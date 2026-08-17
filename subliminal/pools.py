"""Nudge item pools and frame templates.

A pool JSON (data/pools/<name>.json) holds:
  { "name": "animals", "items": [100 strings],
    "frames": [templates with {items}; frames[0] is the base used for collection] }

build_prompt(pool, item_indices, question, frame_idx) assembles
    frame.format(items=", ".join(items)) + " " + question
which for animals frame 0 reproduces the original experiment's prompt exactly.
"""
import json
import os


def load_pool(name_or_path):
    path = name_or_path if os.path.sep in name_or_path or name_or_path.endswith(".json") \
        else os.path.join("data", "pools", f"{name_or_path}.json")
    d = json.load(open(path))
    items = d["items"]
    assert len(items) == len(set(items)), f"pool {d['name']}: duplicate items"
    # 100 is the canonical size (original experiment); animals200 (common
    # animals) and animals1000 (large vocabulary, first 100 = original pool)
    # are the alternative vocabularies.
    assert len(items) in (100, 200, 1000), \
        f"pool {d['name']}: expected 100, 200 or 1000 items, got {len(items)}"
    return d


def build_prompt(pool, idx, question, frame_idx=0):
    items = ", ".join(pool["items"][i] for i in idx)
    frame = pool["frames"][frame_idx].format(items=items)
    return f"{frame} {question}"
