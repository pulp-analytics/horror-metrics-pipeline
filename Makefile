.PHONY: setup test test-fast color-sample clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	@echo "Now: source .venv/bin/activate"

test:
	python3 -m pytest tests/ -v

test-fast:
	python3 -m pytest tests/ -v -m "not slow"

color-sample:
	python3 scripts/01_color_metrics.py

clean:
	rm -rf .venv __pycache__ scripts/__pycache__ scripts/utils/__pycache__ .pytest_cache
