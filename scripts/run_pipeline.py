#!/usr/bin/env python3
"""Run the non-reasoning pipeline (collect -> fit -> extremes) over a set of cells.

For each selected (model, cue, effect) this shells out to the package entry
points in order:

    python -m mhyp.collect  --model M --cue C --effect E [--forced]
    python -m mhyp.fit      --model M --cue C --effect E
    python -m mhyp.extremes --model M --cue C --effect E --mode both

Word-answer effects (e.g. trolley_yn) use the exact forced read; digit-answer
effects (five7, conscious) use the top-k read. GPU required (collect + measure).

    python scripts/run_pipeline.py --model qwen25_7b            # all cues x effects
    python scripts/run_pipeline.py --model qwen25_7b --cue animals_consider --effect five7
    python scripts/run_pipeline.py --all                        # the full 16x4x3 grid
"""
import argparse
import subprocess
import sys

from mhyp import config

# Effects whose answer tokens are words (not single digits) need the exact
# forced-logprob read; digit effects are fine on the top-k read.
FORCED_EFFECTS = {"trolley_yn"}


def run(model, cue, effect, nothink=False, n=None):
    common = ["--model", model, "--cue", cue, "--effect", effect]
    collect = [sys.executable, "-m", "mhyp.collect", *common]
    if effect in FORCED_EFFECTS:
        collect.append("--forced")
    if nothink:
        collect.append("--nothink")
    if n:
        collect += ["--n", str(n)]
    steps = [
        collect,
        [sys.executable, "-m", "mhyp.fit", *common],
        [sys.executable, "-m", "mhyp.extremes", *common, "--mode", "both",
         *(["--nothink"] if nothink else [])],
    ]
    for cmd in steps:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="one model tag (default: all in config.MODELS)")
    ap.add_argument("--cue", help="one cue (default: all)")
    ap.add_argument("--effect", help="one effect (default: all)")
    ap.add_argument("--all", action="store_true", help="run the full grid")
    ap.add_argument("--nothink", action="store_true")
    ap.add_argument("--n", type=int, help="override number of random configs")
    args = ap.parse_args()

    models = [args.model] if args.model else config.MODELS
    cues = [args.cue] if args.cue else list(config.CUES)
    effects = [args.effect] if args.effect else config.EFFECTS
    if not (args.model or args.all):
        ap.error("pass --model TAG (or --all for the whole 16x4x3 grid)")

    for m in models:
        for c in cues:
            for e in effects:
                run(m, c, e, nothink=args.nothink, n=args.n)


if __name__ == "__main__":
    main()
