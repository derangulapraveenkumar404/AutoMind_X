"""Sanity check: does evolution beat plain random search at the SAME number of trained networks?

    python benchmarks/vs_random.py 1 2 3      # seeds; ~1.5 min per seed on one core

Compares the hypervolume (higher = better Pareto front) of the archive found by AutoMind-X against a
random-search archive with an identical training budget. Use it to judge the search honestly.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automind.config import Config  # noqa: E402
from automind.dataset import make_dataset  # noqa: E402
from automind.evaluator import Evaluator, normalised_objectives  # noqa: E402
from automind.evolution import AutoMindX  # noqa: E402
from automind.genome import random_genome  # noqa: E402
from automind.pareto import hypervolume_mc, non_dominated_sort  # noqa: E402


def front_hv(genomes, cfg) -> float:
    front = [genomes[i] for i in non_dominated_sort([g.objectives() for g in genomes])[0]]
    return hypervolume_mc([normalised_objectives(g.accuracy, g.runtime, g.params, cfg) for g in front])


def compare(seed: int) -> tuple[int, float, float]:
    cfg = Config(seed=seed)
    result = AutoMindX(cfg, verbose=False).run()
    budget = result.evaluator.n_trainings
    ev = Evaluator(make_dataset(cfg), cfg)
    rng = np.random.default_rng(seed + 1000)
    rand, seen = [], set()
    while len(rand) < budget:
        g = random_genome(cfg, rng)
        if g.key() not in seen:
            seen.add(g.key())
            rand.append(g)
    ev.evaluate_many(rand)
    return budget, front_hv(result.archive, cfg), front_hv(rand, cfg)


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    print("seed | trainings | evolution HV | random-search HV")
    for s in seeds:
        n, evo, rnd = compare(s)
        print(f"{s:>4} | {n:>9} | {evo:>12.3f} | {rnd:>16.3f}")
