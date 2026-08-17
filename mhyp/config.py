"""Central configuration for the Model Hypnosis experiment suite.

A single place defines the model x cue x effect grid used throughout the paper,
replacing the copies that previously lived in individual scripts. Every runner
and figure script imports the grid from here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CELLS = DATA / "cells"          # per-cell outputs (fetched via scripts/download_data.py)

# ---- the 16 non-reasoning models of Section 3.1 (tags resolve via subliminal/models.py) ----
MODELS = [
    "qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b", "qwen25_72b",
    "qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_32b", "qwen35_9b",
    "gemma2_9b", "gemma4_12b", "llama31_8b", "phi4", "olmo2_7b", "olmo3_7b",
]

# ---- the three forced-choice effects (keys into subliminal/effects.py::EFFECTS) ----
EFFECTS = ["five7", "trolley_yn", "conscious"]

# ---- the four cue families ----
# kind="pool": L distinct items drawn from a shared pool.
# kind="bank": L slots, each with O admissible fragments (paraphrases / typos / JSON fields).
CUES = {
    "animals_consider": {"kind": "pool", "pool": "animals_consider", "L": 10},
    "phrasing_L20_O10": {"kind": "bank", "bank": "sentences20x10",    "L": 20, "O": 10},
    "jsonblob":         {"kind": "bank", "bank": "json12x6",          "L": 12, "O": 6},
    "typos":            {"kind": "bank", "bank": "sentences20_typos", "L": 20, "O": 6},
}

# Random prompt configurations evaluated per cell before the additive fit.
N_RANDOM = 12_000            # non-reasoning cells (Section 3.1)
N_RANDOM_REASONING = 20_000  # open-weight reasoning cells (Section 3.2)


def cell_dir(tag: str, cue: str, effect: str) -> Path:
    """data/cells/<tag>/<cue>_<effect>/ -- the standard per-cell directory."""
    return CELLS / tag / f"{cue}_{effect}"


def is_pool(cue: str) -> bool:
    return CUES[cue]["kind"] == "pool"


def grid():
    """All (model, cue, effect) triples in the non-reasoning suite."""
    return [(m, c, e) for m in MODELS for c in CUES for e in EFFECTS]
