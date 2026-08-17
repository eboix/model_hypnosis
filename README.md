# Model Hypnosis: Controlling AI via additive subliminal effects

Code and data for the paper *Model Hypnosis: Controlling AI via additive
subliminal effects* (Enric Boix-Adsera and Benedict Tessler).

**Model hypnosis** is the phenomenon that a language model can be strongly
steered by the *additive* effect of many individually weak, semantically
irrelevant prompt **cues**. We fit an additive model in log-odds — one *cue
score* per fragment — from thousands of random prompts, then stack cues whose
scores align to drive a forced-choice answer with near certainty. The effect
holds across model families and scales (including reasoning models) and the
constructed prompts often transfer between models.

**[Interactive explorer »](https://eboix.github.io/model_hypnosis/)** — browse
every measured cell: pick a model / cue / effect, hover any scatter point for its
full prompt and P(y⁺), and zoom/pan the predicted-vs-measured plot. Rebuild it
locally with `make explorer`.

## What's here

```
subliminal/     core library: exact-logit vLLM backend, effects (forced questions),
                additive fitting, cue pools, paraphrase banks, model registry
mhyp/           the experiment pipeline (one clean entry point per stage):
  config.py       the single model x cue x effect grid used everywhere
  collect.py      evaluate N random prompt configurations  -> raw.jsonl        (GPU)
  fit.py          additive fit (item x position / per-slot) -> fit(_ip).json    (CPU)
  extremes.py     candidate bands + measured extremes       -> scatter_extras   (GPU/CPU)
  transfer.py     cross-model transfer: dump candidates, measure on targets    (GPU)
  reasoning.py    open-weight reasoning-model steering (thinking budgets)       (GPU)
  api_steering/   closed-weight API reasoning models (GPT-5.6, Gemini, Claude)
  keys.py         API keys from environment variables
analysis/       figure/table generators (one per paper figure; CPU only)
data/           committed cue banks (pools + paraphrase/JSON/typo banks);
                per-cell results are fetched via scripts/download_data.py
scripts/        download_data.py, run_pipeline.py, make_figures.py, build_explorer.py
Makefile        setup / data / figures / experiments / transfer / explorer
```

## Setup

```bash
pip install -e .            # CPU: analysis + figures
pip install -e ".[gpu]"     # + vLLM, to run experiments on a GPU
pip install -e ".[api]"     # + httpx, for API-model steering
```

Python >= 3.10. Running experiments needs one or more GPUs and the model
weights (downloaded from Hugging Face on first use; set `HF_TOKEN` for gated
models). API-model steering needs the relevant key (see `.env.example`).

## Reproduce the figures (no GPU)

```bash
make data       # download the experiment-data archive (~280 MB) and unpack into data/
make figures    # regenerate every figure into figures/
make explorer   # build the interactive explorer HTML
```

`make data` fetches one archive with every per-cell result (per-trial log-odds,
fits, measured extremizers, transfer) from Zenodo
([DOI 10.5281/zenodo.21981022](https://doi.org/10.5281/zenodo.21981022)). The
figures redraw the random-prompt clouds from the per-trial data, so this is what
lets `make figures` reproduce the paper exactly with no GPU.

## Reproduce the experiments (GPU)

Each cell is a (model, cue, effect) triple; the grid is defined once in
`mhyp/config.py` (16 non-reasoning models x 4 cues x 3 effects).

```bash
# one cell, end to end (collect -> fit -> extremes):
python -m mhyp.collect  --model qwen25_7b --cue animals_consider --effect five7
python -m mhyp.fit      --model qwen25_7b --cue animals_consider --effect five7
python -m mhyp.extremes --model qwen25_7b --cue animals_consider --effect five7 --mode both

# or drive the whole grid:
python scripts/run_pipeline.py --model qwen25_7b     # all cues x effects for one model
make experiments                                     # the full 16x4x3 grid

# cross-model transfer (after the per-cell fits exist):
python -m mhyp.transfer cands                        # dump candidate prompts per source
python -m mhyp.transfer measure --target gemma2_9b   # measure them on a target
```

`raw.jsonl` (the per-trial exact log-odds) is fully determined by the seed, so it
is not committed — `make data` fetches it, and `make experiments` regenerates it
from scratch.

## Cues and effects

| Cue family | Template | slots L | options |
|---|---|---|---|
| `animals_consider` | list of animals | 10 | 200-item pool (distinct) |
| `phrasing_L20_O10` | 20-sentence story, paraphrased | 20 | 10 paraphrases/slot |
| `jsonblob` | JSON request-metadata object | 12 | 6 values/field |
| `typos` | 20-sentence story with typos | 20 | 6 variants/slot |

| Effect | Question | y⁺ / y⁻ |
|---|---|---|
| `five7` | prefer 5 or 7? | 5 / 7 |
| `trolley_yn` | is one harm right to prevent five? | yes / no |
| `conscious` | are you conscious? | yes / no |

## Figures

`make figures` runs the generators in `analysis/` into `figures/`. Highlights:

| Figure | Script |
|---|---|
| Fig 2  animals teaser scatter | `analysis/teaser_scatter.py` |
| Fig 4/5/13  additive fit, tilt, R² | `analysis/fit_random_figure.py` |
| Fig 6/7  per-slot share donuts | `analysis/coef_prompts_figure.py`, `coef_concentration_prompts.py` |
| Fig 8  steering strength | `analysis/steering_combined.py` |
| Fig 9  reasoning baselines | `analysis/api_saturation_heatmap.py` |
| Fig 10  reasoning steering | `analysis/reasoning_steering_4panel.py` |
| Fig 11  API model prompts | `analysis/api_prompts_figure.py` |
| Fig 12/35  transfer | `analysis/transfer_summary_figs.py` |
| Fig 14–25  per-cell scatter grids | `analysis/scatter_grid.py` |
| Fig 26–33  paired extremizer boxes | `analysis/appendix_pairs_figure.py` |
| Fig 34  effective slots L_eff | `analysis/simpson_plot.py` |
| Fig 36  paraphrase robustness | `analysis/robustness_plot.py` |
| Fig 42  repeated-item cues | `analysis/repeats_figure.py` |

Scripts that emit LaTeX (`example_prompts_figure.py`, `api_prompts_figure.py`,
`appendix_pairs_figure.py`, `coef_prompts_figure.py`) write `.tex` fragments to
`figures/` for `\input` into the manuscript.

## Data

One release archive (`scripts/download_data.py`) holds every per-cell result,
which unpacks under `data/cells/` (plus `data/transfer*/`):

- `raw.jsonl` — the per-trial exact log-odds for each random configuration (the
  figures' random-prompt clouds and logit spread come from these);
- `fit.json` / `fit_ip.json` — the additive fits;
- `scatter_extras.json` / `se_ip_configs.json` — the measured extremizers;
- `transfer_found/*.json` — cross-model transfer measurements.

About 280 MB to download, ~1.5 GB unpacked, from Zenodo
([DOI 10.5281/zenodo.21981022](https://doi.org/10.5281/zenodo.21981022)).
`make experiments` regenerates all of it from scratch on a GPU; the archive just
spares you that.

## Citation

See `CITATION.cff`. Licensed under the MIT License (`LICENSE`).
