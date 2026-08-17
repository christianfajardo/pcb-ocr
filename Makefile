.PHONY: up down test lint format clean health logs

up:
	bash scripts/start_services.sh

down:
	bash scripts/stop_services.sh

test:
	bash scripts/run_tests.sh

test-local:
	PYTHONPATH=. python scripts/local_test.py

lint:
	ruff check .

format:
	ruff format .

clean:
	docker compose down --rmi all --volumes
	find . -type d -name __pycache__ -exec rm -rf {} +

health:
	python scripts/health_check.py

logs:
	docker compose logs -f

benchmark:
	PYTHONPATH=. python scripts/benchmark.py