"""AutoMind-X entry point.

python main.py --quick       10-second demo
python main.py               full run
python main.py --help        all options
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time

from .config import MAX_SEED, RUNTIME_METRICS, SELECTION_STRATEGIES, Config
from .evolution import AutoMindX, Result
from .explain import explain, holdout_report, select_final
from .genome import Genome
from .report import write_html
from .visualize import make_all


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _seed(text: str) -> int:
    value = int(text)
    if not 0 <= value <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"must be between 0 and {MAX_SEED}")
    return value


def _finite_positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite number > 0")
    return value


def _safe_outdir(text: str) -> str:
    """Resolve the output directory and refuse anything that is an existing non-directory."""
    path = os.path.abspath(os.path.expanduser(text))
    if os.path.exists(path) and not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"{text!r} exists and is not a directory")
    return path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="AutoMind-X: explainable, self-adaptive, multi-objective neural architecture search"
    )
    ap.add_argument("--quick", action="store_true", help="small fast demo")
    ap.add_argument("--generations", type=_positive_int, help="number of generations")
    ap.add_argument("--pop", type=_positive_int, help="population size (>= 2)")
    ap.add_argument("--seed", type=_seed, help="random seed (default 42)")
    ap.add_argument("--outdir", type=_safe_outdir, default="outputs", help="output directory (default: outputs)")
    ap.add_argument(
        "--runtime-metric",
        choices=RUNTIME_METRICS,
        help="'proxy' = deterministic FLOPs estimate (default), 'wallclock' = measured seconds",
    )
    ap.add_argument(
        "--select",
        choices=SELECTION_STRATEGIES,
        dest="selection",
        help="final model choice: 'balanced' weighted fitness or 'knee' ideal-point distance",
    )
    ap.add_argument("--max-params", type=_positive_int, help="deployment budget: max model parameters")
    ap.add_argument(
        "--max-cost", type=_finite_positive_float, help="deployment budget: max training cost (pseudo-seconds)"
    )
    ap.add_argument("--n-jobs", type=int, help="parallel workers for training (-1 = all cores, 1 = serial)")
    ap.add_argument("--no-plots", action="store_true", help="skip figure generation")
    ap.add_argument("--json", action="store_true", help="also write summary.json (machine-readable)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-generation progress")
    return ap


def build_config(a: argparse.Namespace) -> Config:
    cfg = Config(outdir=a.outdir)
    if a.quick:
        cfg.pop_size, cfg.generations, cfg.n_samples, cfg.max_iter = 8, 5, 1000, 40
    for attr, val in (
        ("generations", a.generations),
        ("pop_size", a.pop),
        ("seed", a.seed),
        ("runtime_metric", a.runtime_metric),
        ("selection", a.selection),
        ("n_jobs", a.n_jobs),
        ("max_params", a.max_params),
        ("max_cost", a.max_cost),
    ):
        if val is not None:
            setattr(cfg, attr, val)
    cfg.validate()
    return cfg


def write_outputs(result: Result, cfg: Config, report: str, chosen: Genome, hold: tuple, write_json: bool) -> None:
    """Write the text report, the architecture CSV and (optionally) summary.json."""
    os.makedirs(cfg.outdir, exist_ok=True)
    with open(os.path.join(cfg.outdir, "explanation.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(cfg.outdir, "all_architectures.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "uid",
                "generation",
                "layers",
                "activation",
                "lr",
                "accuracy",
                "runtime_s",
                "wall_time_s",
                "params",
                "fitness",
            ]
        )
        for g in result.archive:
            w.writerow(
                [
                    g.uid,
                    g.birth_gen,
                    "-".join(map(str, g.layers)),
                    g.activation,
                    g.lr,
                    f"{g.accuracy:.4f}",
                    f"{g.runtime:.6f}",
                    f"{g.wall_time:.6f}",
                    g.params,
                    f"{g.fitness:.4f}",
                ]
            )
    if write_json:
        test_acc, test_rt, test_params = hold
        summary = dict(
            seed=cfg.seed,
            budget=dict(max_params=cfg.max_params, max_cost=cfg.max_cost),
            runtime_metric=cfg.runtime_metric,
            selection=cfg.selection,
            selected=dict(
                layers=chosen.layers,
                activation=chosen.activation,
                lr=chosen.lr,
                val_accuracy=chosen.accuracy,
                params=chosen.params,
                found_in_generation=chosen.birth_gen,
            ),
            holdout=dict(test_accuracy=test_acc, cost_s=test_rt, params=test_params),
            trainings=result.evaluator.n_trainings,
            cache_hits=result.evaluator.cache_hits,
            final_hypervolume=result.history[-1]["hypervolume"],
        )
        with open(os.path.join(cfg.outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        cfg = build_config(a)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    t0 = time.time()
    try:
        result = AutoMindX(cfg, verbose=not a.quiet).run()
        chosen = select_final(result)[0]
        hold = result.evaluator.holdout(chosen)
        report = explain(result) + "\n" + holdout_report(result, chosen)
        write_outputs(result, cfg, report, chosen, hold, a.json)
        paths = [] if a.no_plots else [*make_all(result, cfg.outdir), write_html(result, cfg.outdir)]
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        return 130
    except OSError as exc:  # unwritable output directory, disk full, ...
        print(f"error: cannot write results to {cfg.outdir!r}: {exc.strerror or exc}", file=sys.stderr)
        return 1

    print("\n" + report)
    print(
        f"\nSaved: {', '.join(os.path.basename(p) for p in paths)}{', ' if paths else ''}explanation.txt, "
        f"all_architectures.csv -> {cfg.outdir}/   (total {time.time() - t0:.1f}s, "
        f"{result.evaluator.n_trainings} networks trained, {result.evaluator.cache_hits} cache hits)"
    )
    return 0
