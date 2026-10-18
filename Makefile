PY ?= .venv/bin/python

.PHONY: install test forecast planning computation pareto online network figures tables report all

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install chronos-forecasting

test:
	$(PY) -m pytest -q

# 1. forecasting backtest (~15 min), foundation-model baseline (~1 h on CPU), joint scenarios
forecast:
	$(PY) -m experiments.e01_forecast_backtest
	$(PY) -m experiments.e01b_chronos small base
	$(PY) -m experiments.e02_scenarios

# 2. planning: value of forecasts (31 days), SAA bounds / VSS / EVPI, flexibility sensitivity
planning:
	$(PY) -m experiments.e05_value_of_forecasts
	$(PY) -m experiments.e04_saa
	$(PY) -m experiments.e09_sensitivity

# 3. computational study (run alone: wall-clock timings)
computation:
	$(PY) -m experiments.e03_scaling a b c

pareto:
	$(PY) -m experiments.e06_pareto

online:
	$(PY) -m experiments.e07_online

network:
	$(PY) -m experiments.e08_network_design

figures:
	$(PY) -m experiments.figures

tables:
	$(PY) -m experiments.tables

report: figures tables
	cd report && latexmk -pdf -interaction=nonstopmode main.tex

all: forecast planning computation pareto online network report
