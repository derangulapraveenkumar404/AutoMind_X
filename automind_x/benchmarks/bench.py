"""Micro-benchmark: serial vs parallel evaluation and cache effectiveness.

    python benchmarks/bench.py            # prints a small table
Results depend on core count; on a single-core machine both columns are about equal.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automind.config import Config  # noqa: E402
from automind.dataset import make_dataset  # noqa: E402
from automind.evaluator import Evaluator  # noqa: E402
from automind.genome import random_genome  # noqa: E402
from automind.pareto import non_dominated_sort  # noqa: E402


def time_batch(n_jobs: int, n: int = 16) -> float:
    cfg = Config(n_jobs=n_jobs, max_iter=40, n_samples=1500)
    ev = Evaluator(make_dataset(cfg), cfg)
    rng = np.random.default_rng(0)
    genomes = [random_genome(cfg, rng) for _ in range(n)]
    t0 = time.perf_counter()
    ev.evaluate_many(genomes)
    return time.perf_counter() - t0


def time_sort(n: int = 400) -> float:
    objs = np.random.default_rng(0).random((n, 3)).tolist()
    t0 = time.perf_counter()
    non_dominated_sort(objs)
    return time.perf_counter() - t0


if __name__ == "__main__":
    print(f"cpu cores            : {os.cpu_count()}")
    print(f"train 16 nets serial : {time_batch(1):.2f}s")
    print(f"train 16 nets n_jobs=-1: {time_batch(-1):.2f}s")
    print(f"non-dominated sort of 400 points: {time_sort() * 1000:.1f} ms")
