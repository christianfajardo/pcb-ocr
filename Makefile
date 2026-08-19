# Prefer the project-local venv so these targets work without the caller
# activating it first; falls back to python3 if .venv isn't set up yet.
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: up down test test-local lint format clean health logs benchmark

up:
	bash scripts/start_services.sh

down:
	bash scripts/stop_services.sh

test:
	bash scripts/run_tests.sh

test-local:
	PYTHONPATH=. $(PYTHON) scripts/local_test.py

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

clean:
	docker compose down --rmi all --volumes
	find . -type d -name __pycache__ -exec rm -rf {} +

health:
	$(PYTHON) scripts/health_check.py

logs:
	docker compose logs -f

benchmark:
	set -a; [ -f .env ] && . .env; set +a; PYTHONPATH=. $(PYTHON) scripts/benchmark.py