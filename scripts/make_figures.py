#!/usr/bin/env python3
"""Regenerate every paper figure into $MHYP_FIGDIR (default: figures/).

Runs the canonical generators in analysis/. CPU only -- it reads the committed /
downloaded per-cell results (run `make data` first). Each generator is run
independently; a failure (e.g. a cell whose data was not downloaded) is reported
and the rest continue.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mhyp import config  # noqa: E402

PY = sys.executable
ANALYSIS = ROOT / "analysis"
OUT = os.environ.get("MHYP_FIGDIR", "figures")

# Generators that take no arguments.
BARE = [
    "teaser_scatter.py",           # Fig 2
    "fit_random_figure.py",        # Fig 4, 5, 13
    "coef_prompts_figure.py",      # Fig 6/7 (tex)
    "coef_concentration_prompts.py",
    "steering_combined.py",        # Fig 8
    "api_saturation_heatmap.py",   # Fig 9
    "reasoning_steering_4panel.py",# Fig 10
    "api_prompts_figure.py",       # Fig 11 (tex)
    "transfer_summary_figs.py",    # Fig 12, 35 + sec-4 panels
    "appendix_pairs_figure.py",    # Fig 26-33 (tex)
    "simpson_plot.py",             # Fig 34
    "robustness_plot.py",          # Fig 36
    "repeats_figure.py",           # Fig 42
    "example_prompts_figure.py",   # Fig 3 (tex)
    "transfer_matrix.py",          # per-cue transfer heatmaps
    "steering_heatmap_ranges.py",
    "steering_range_plot.py",
    "scattertilt_appendix.py",
    "app_nudge_details.py",
    "cell_counts_table.py",
]


def run(cmd, label):
    print(f"\n=== {label} ===", flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  !! {label} failed (exit {e.returncode}) -- continuing", flush=True)
        return False


def main():
    os.makedirs(ROOT / OUT, exist_ok=True)
    ok = fail = 0
    for script in BARE:
        (ok := ok + 1) if run([PY, str(ANALYSIS / script)], script) else (fail := fail + 1)

    # Fig 14-25: one 16-model scatter grid per cue x effect.
    models = ",".join(config.MODELS)
    for cue in config.CUES:
        for eff in config.EFFECTS:
            label = f"scatter_grid {cue} {eff}"
            cmd = [PY, str(ANALYSIS / "scatter_grid.py"),
                   "--nudge", cue, "--eff", eff, "--models", models,
                   "--out", f"grid_final_{cue}_{eff}.png"]
            (ok := ok + 1) if run(cmd, label) else (fail := fail + 1)

    print(f"\nfigures -> {OUT}/   ({ok} ok, {fail} failed)")
    if fail:
        print("failed generators usually mean their cell data was not downloaded "
              "(`make data`) or needs a GPU stage first.")


if __name__ == "__main__":
    main()
