# Changelog

## 1.2.0
### Fixed (audit)
- `pip install .` / wheel build failed (no build-system, ambiguous package discovery) -> fixed; `automind-x` console script added; CLI moved to `automind/cli.py`.
- Negative or oversized `--seed` crashed with a traceback -> validated (0..2^32-1), exit code 2.
- Unwritable `--outdir` crashed with a traceback -> clean message, exit code 1; Ctrl-C exits 130.
- `--max-cost inf` was accepted -> must be finite; resource guards on population, generations, samples, iterations.

## 1.1.0
### Added
- Deterministic FLOPs-based runtime proxy (`--runtime-metric proxy`, default) -> reproducible runs.
- Parallel, deduplicated batch evaluation of offspring (`--n-jobs`), with serial fallback.
- Knee-point selection (`--select knee`) and deployment budgets (`--max-params`, `--max-cost`).
- Accessible `report.html`, `ALT_TEXT.md`, colour-blind-safe plots, `summary.json` (`--json`).
- Config validation, argparse validation, exit codes.
- Property-based tests (Hypothesis), CLI/end-to-end tests (75 tests, ~99% coverage).
- `pyproject.toml` (ruff, mypy, pytest), CI (lint, types, security scan, tests), Dockerfile, Dependabot.
### Changed
- NumPy-vectorised dominance computation; memoised hold-out/baseline training.
- CSV now stores runtime with 6 decimals so the Pareto front can be recomputed exactly from it.
- `Genome` uses identity equality; logging replaces `print`.

## 1.0.0
- Initial NSGA-II + self-adaptive mutation architecture search with explanation report.
