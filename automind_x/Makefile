.PHONY: check lint types security test demo bench
check: lint types security test
lint:
	ruff check . && ruff format --check .
types:
	mypy automind main.py
security:
	bandit -q -r automind main.py
test:
	python -m pytest --cov=automind --cov-fail-under=90
demo:
	python main.py --quick --json
bench:
	python benchmarks/bench.py
