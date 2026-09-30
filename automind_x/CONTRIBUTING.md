# Contributing

```bash
pip install -e ".[dev]"      # or: pip install -r requirements-dev.txt
make check                   # ruff + mypy + bandit + tests (same as CI)
```

* Keep functions small and typed; add a test for every behaviour change.
* Search-space extensions: add a gene in `genome.py`, a mutation op in `operators.py`, and a test that
  `test_operators_keep_genomes_valid` still holds.
* Do not commit secrets or generated caches (see `.gitignore`, `SECURITY.md`).
