"""Fitness evaluation: train a candidate network and measure the 3 objectives."""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Iterable

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

log = logging.getLogger(__name__)

# (accuracy, runtime_objective, params, wall_time)
Measurement = tuple[float, float, int, float]


def count_params(n_in: int, layers, n_out: int) -> int:
    """Weights + biases of a fully connected network."""
    sizes = [n_in, *layers, n_out]
    return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:], strict=True)))


def proxy_runtime(params: int, n_train: int, epochs: int, flops_per_s: float) -> float:
    """Deterministic training-cost estimate in pseudo-seconds.

    Forward + backward pass ~ 6 FLOPs per parameter per sample, times the epochs actually run.
    Being hardware independent, it makes the whole search reproducible.
    """
    return 6.0 * params * n_train * max(1, epochs) / flops_per_s


def normalised_objectives(acc, runtime, params, cfg):
    """All three objectives scaled to [0, 1], lower = better."""
    return (1.0 - acc, min(1.0, runtime / cfg.time_ref), min(1.0, params / cfg.params_ref))


def fitness_terms(acc, runtime, params, cfg):
    """Return (accuracy reward, time penalty, size penalty)."""
    return (
        cfg.w_acc * acc,
        cfg.w_time * min(1.0, runtime / cfg.time_ref),
        cfg.w_params * min(1.0, params / cfg.params_ref),
    )


def scalar_fitness(acc, runtime, params, cfg) -> float:
    reward, t_pen, p_pen = fitness_terms(acc, runtime, params, cfg)
    return reward - t_pen - p_pen


def _train_and_score(spec, data, cfg, eval_split: str = "val") -> Measurement:
    """Train one network. Module-level (picklable) so it can run in worker processes."""
    layers, activation, lr = spec
    model = MLPClassifier(
        hidden_layer_sizes=tuple(layers),
        activation=activation,
        learning_rate_init=lr,
        max_iter=cfg.max_iter,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=cfg.seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        t0 = time.perf_counter()
        model.fit(data.X_train, data.y_train)
        wall = time.perf_counter() - t0
    X, y = (data.X_val, data.y_val) if eval_split == "val" else (data.X_test, data.y_test)
    acc = float(model.score(X, y))
    params = count_params(data.n_features, layers, data.n_classes)
    if cfg.runtime_metric == "proxy":
        runtime = proxy_runtime(params, len(data.y_train), int(model.n_iter_), cfg.proxy_flops_per_s)
    else:
        runtime = wall
    return acc, runtime, params, wall


# Below this batch size, process start-up costs more than the parallel speed-up saves.
MIN_PARALLEL_BATCH = 4


class Evaluator:
    """Trains candidates, caches results by architecture, and evaluates batches in parallel."""

    def __init__(self, data, cfg):
        self.data, self.cfg = data, cfg
        self._cache: dict = {}
        self._holdout_cache: dict = {}
        self.n_trainings = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------ single / batch evaluation
    def _apply(self, g, m: Measurement) -> None:
        """Copy a cached measurement onto a genome and compute its scalar fitness."""
        g.accuracy, g.runtime, g.params, g.wall_time = m
        g.fitness = scalar_fitness(g.accuracy, g.runtime, g.params, self.cfg)

    def evaluate(self, g):
        """Fill g.accuracy / runtime / params / fitness. Identical architectures are cached."""
        return self.evaluate_many([g])[0]

    def evaluate_many(self, genomes: Iterable) -> list:
        """Evaluate a batch; each *unique unseen* architecture is trained once, in parallel if allowed."""
        genomes = list(genomes)
        todo = {}
        for g in genomes:
            if g.key() not in self._cache and g.key() not in todo:
                todo[g.key()] = (tuple(g.layers), g.activation, g.lr)
        results = self._train_batch(list(todo.values()))
        for key, m in zip(todo.keys(), results, strict=True):
            self._cache[key] = m
            self.n_trainings += 1
        self.cache_hits += len(genomes) - len(todo)
        for g in genomes:
            self._apply(g, self._cache[g.key()])
        return genomes

    def _train_batch(self, specs: list) -> list[Measurement]:
        if not specs:
            return []
        jobs = self.cfg.n_jobs
        # Parallel wall-clock timings would be distorted by CPU contention -> serial in that mode.
        if len(specs) >= MIN_PARALLEL_BATCH and jobs != 1 and self.cfg.runtime_metric == "proxy":
            try:
                from joblib import Parallel, delayed

                return list(Parallel(n_jobs=jobs)(delayed(_train_and_score)(s, self.data, self.cfg) for s in specs))
            except Exception as exc:  # sandbox without multiprocessing, etc. -> degrade gracefully
                log.warning("parallel evaluation unavailable (%s); falling back to serial", exc)
        return [_train_and_score(s, self.data, self.cfg) for s in specs]

    # ------------------------------------------------------------------ hold-out + baseline
    def holdout(self, g):
        """Retrain and score on the untouched TEST split -> (test_acc, runtime, params). Memoised."""
        key = g.key()
        if key not in self._holdout_cache:
            acc, runtime, params, _ = _train_and_score(
                (tuple(g.layers), g.activation, g.lr), self.data, self.cfg, eval_split="test"
            )
            self._holdout_cache[key] = (acc, runtime, params)
        return self._holdout_cache[key]

    def baseline(self):
        """A plain 'default' MLP (100,) for comparison."""
        from .genome import Genome

        base = Genome(layers=[100], activation="relu", lr=0.001, mutation_rate=0.0)
        return base, self.holdout(base)
