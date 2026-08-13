.PHONY: setup test color-sample clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	@echo "Now: source .venv/bin/activate"

test:
	python3 -m pytest tests/ -v

color-sample:
	python3 scripts/01_color_metrics.py
	python3 scripts/02_aggregate_and_checkpoint.py --chart data/sample_output/darkness_curve.png

clean:
	rm -rf .venv __pycache__ scripts/__pycache__ scripts/utils/__pycache__ .pytest_cache
	rm -f data/sample_output/yearly.json data/sample_output/hue_river.json data/sample_output/darkness_curve.png
