"""Central configuration. Every knob of the system lives here, and is validated on creation."""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_SEED = 2**32 - 1  # numpy / scikit-learn accept seeds in [0, 2**32 - 1]
MAX_POP, MAX_GENERATIONS, MAX_SAMPLES, MAX_ITER = 1000, 1000, 200_000, 2000  # resource guards
RUNTIME_METRICS = ("proxy", "wallclock")
SELECTION_STRATEGIES = ("balanced", "knee")


@dataclass
class Config:
    # ---- reproducibility ----
    seed: int = 42

    # ---- synthetic dataset (scikit-learn make_classification) ----
    n_samples: int = 2500
    n_features: int = 20
    n_informative: int = 10
    n_redundant: int = 4
    n_classes: int = 3
    class_sep: float = 0.9
    flip_y: float = 0.03

    # ---- search space ----
    min_layers: int = 1
    max_layers: int = 4
    neuron_choices: tuple = (4, 8, 16, 32, 64, 128)
    activations: tuple = ("relu", "tanh", "logistic")
    lr_choices: tuple = (0.001, 0.003, 0.01, 0.03)

    # ---- training budget for each candidate network ----
    max_iter: int = 60

    # ---- runtime objective ----
    # "proxy"     : deterministic FLOPs-based estimate -> fully reproducible runs (default)
    # "wallclock" : measured seconds -> realistic but noisy
    runtime_metric: str = "proxy"
    proxy_flops_per_s: float = 1.0e9  # calibration of the FLOPs -> pseudo-seconds conversion

    # ---- efficiency ----
    n_jobs: int = -1  # parallel training of offspring (-1 = all cores, 1 = serial)

    # ---- genetic algorithm ----
    pop_size: int = 14
    generations: int = 12
    crossover_rate: float = 0.85
    tournament_size: int = 2

    # ---- self-adaptive mutation + meta-evolution ----
    init_mutation_rate: float = 0.35  # starting value of every individual's own gene
    mut_min: float = 0.05
    mut_max: float = 0.90
    meta_tau: float = 0.30  # step size for evolving the mutation-rate gene
    mult_min: float = 0.40  # bounds of the population-level multiplier
    mult_max: float = 2.50
    improve_eps: float = 1e-3  # what counts as "the population improved"

    # ---- scalar fitness (used for tracking, adaptation and final pick) ----
    w_acc: float = 0.60
    w_time: float = 0.20
    w_params: float = 0.20
    time_ref: float = 0.30  # seconds that count as "maximally slow"
    params_ref: float = 6000  # parameter count that counts as "maximally big"

    # ---- final model selection ----
    selection: str = "balanced"  # "balanced" (weighted fitness) or "knee" (closest to ideal point)
    max_params: int | None = None  # deployment budget: model may have at most this many parameters
    max_cost: float | None = None  # deployment budget: training cost (same unit as runtime) upper bound

    # ---- output ----
    outdir: str = "outputs"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Fail fast with a clear message instead of crashing deep inside the search."""

        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"Invalid Config: {msg}")

        need(0 <= self.seed <= MAX_SEED, f"seed must be in [0, {MAX_SEED}]")
        need(100 <= self.n_samples <= MAX_SAMPLES, f"n_samples must be in [100, {MAX_SAMPLES}]")
        need(self.n_classes >= 2, "n_classes must be >= 2")
        need(
            self.n_informative + self.n_redundant <= self.n_features,
            "n_informative + n_redundant must be <= n_features",
        )
        need(self.n_informative >= 1, "n_informative must be >= 1")
        need(2**self.n_informative >= self.n_classes, "n_informative too small for n_classes")
        need(1 <= self.min_layers <= self.max_layers, "1 <= min_layers <= max_layers required")
        need(
            len(self.neuron_choices) > 0 and all(int(n) > 0 for n in self.neuron_choices),
            "neuron_choices must be positive integers",
        )
        need(len(self.activations) > 1, "need at least two activations")
        need(len(self.lr_choices) > 0 and all(x > 0 for x in self.lr_choices), "lr_choices must be > 0")
        need(1 <= self.max_iter <= MAX_ITER, f"max_iter must be in [1, {MAX_ITER}]")
        need(2 <= self.pop_size <= MAX_POP, f"pop_size must be in [2, {MAX_POP}]")
        need(1 <= self.generations <= MAX_GENERATIONS, f"generations must be in [1, {MAX_GENERATIONS}]")
        need(0.0 <= self.crossover_rate <= 1.0, "crossover_rate must be in [0, 1]")
        need(self.tournament_size >= 1, "tournament_size must be >= 1")
        need(0.0 < self.mut_min <= self.mut_max <= 1.0, "0 < mut_min <= mut_max <= 1 required")
        need(0.0 < self.mult_min <= self.mult_max, "0 < mult_min <= mult_max required")
        need(min(self.w_acc, self.w_time, self.w_params) >= 0.0, "fitness weights must be >= 0")
        need(self.time_ref > 0 and self.params_ref > 0, "time_ref and params_ref must be > 0")
        need(self.proxy_flops_per_s > 0, "proxy_flops_per_s must be > 0")
        need(self.runtime_metric in RUNTIME_METRICS, f"runtime_metric must be one of {RUNTIME_METRICS}")
        need(self.selection in SELECTION_STRATEGIES, f"selection must be one of {SELECTION_STRATEGIES}")
        need(self.max_params is None or self.max_params > 0, "max_params must be > 0")
        need(
            self.max_cost is None or (math.isfinite(self.max_cost) and self.max_cost > 0),
            "max_cost must be a finite number > 0",
        )
        need(self.n_jobs != 0, "n_jobs must be non-zero (-1 = all cores)")
