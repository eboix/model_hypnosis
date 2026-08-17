# Model Hypnosis -- reproduce the paper's experiments and figures.
#
# Typical use:
#   make setup && make data && make figures      # regenerate figures, no GPU
#   make setup-gpu && make experiments           # rerun evals from scratch (GPU)
PY     ?= python
FIGDIR ?= figures
export MHYP_FIGDIR = $(FIGDIR)

.PHONY: help setup setup-gpu setup-api data figures explorer \
        experiments transfer clean

help:
	@echo "make setup        install the package (CPU: analysis + figures)"
	@echo "make setup-gpu    install with vLLM, for running experiments"
	@echo "make setup-api    install httpx, for API-model steering"
	@echo "make data         download the experiment-data archive (~280 MB, ~1.5 GB unpacked)"
	@echo "make figures      regenerate every paper figure into $(FIGDIR)/ (no GPU)"
	@echo "make explorer     build data/explorer.json + the interactive HTML"
	@echo "make experiments  rerun the full 16x4x3 non-reasoning grid (GPU)"
	@echo "make transfer     dump transfer candidates + measure on targets (GPU)"

setup:
	$(PY) -m pip install -e .

setup-gpu:
	$(PY) -m pip install -e ".[gpu]"

setup-api:
	$(PY) -m pip install -e ".[api]"

data:
	$(PY) scripts/download_data.py

figures:
	$(PY) scripts/make_figures.py

explorer:
	$(PY) analysis/explorer_data.py
	$(PY) scripts/build_explorer.py

# GPU. Runs collect -> fit -> extremes for every (model, cue, effect).
experiments:
	$(PY) scripts/run_pipeline.py --all

# GPU. Dump source candidates (CPU), then measure on each target model.
transfer:
	$(PY) -m mhyp.transfer cands
	@echo "next, per target:  $(PY) -m mhyp.transfer measure --target <tag>"

clean:
	rm -rf $(FIGDIR) build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
