"""Collect random-configuration exact-logit reads for one (model, cue, effect).

For a given cell this evaluates N random prompt configurations and writes the
per-trial records to ``raw.jsonl`` under ``config.cell_dir(tag, cue, effect)``.
Each line is ``{"ch": [...]}`` plus ``"p"`` and ``"l"`` when the read succeeds,
where ``ch`` is the ordered configuration (pool: item indices; bank: per-slot
fragment choices) and ``l`` is the exact forced-choice log-odds
log P(tok_a) - log P(tok_b). Downstream fitting/plotting tools consume this
schema, so it is kept byte-for-byte compatible with the source producers.

Two cue families are merged here and selected via ``config.is_pool(cue)``:
  * pool cues  -- N random DISTINCT-item lists drawn from a shared item pool
                  (ported from data/_pool_gen.py);
  * bank cues  -- N random per-slot fragment choices from a paraphrase/typo/
                  JSON bank (ported from data/_phrasing_gen.py).

Reads use the exact prompt-logprob path (read_batch_forced) with --forced,
otherwise the top-k path (read_batch). Runs are resumable: existing lines in
raw.jsonl are counted and skipped, and the RNG is fully determined by --seed so
the resumed suffix is identical to an uninterrupted run.

Requires a GPU: LocalModel loads the weights under vLLM.

  python -m mhyp.collect --model qwen25_7b --cue animals_consider \
      --effect five7 [--n 12000] [--forced] [--nothink] [--seed 0]
"""
import argparse
import json
import os
import random

import numpy as np  # noqa: F401  (kept available; not required for raw.jsonl)

from subliminal.backend import LocalModel
from subliminal.effects import EFFECTS
from subliminal.models import resolve
from subliminal.pools import load_pool, build_prompt

from mhyp import config

CHUNK = 250  # trials per vLLM batch (matches both source producers)


def _count_done(path):
    return sum(1 for _ in open(path)) if os.path.exists(path) else 0


def _make_pool_trials(cue, seed, n):
    """N random distinct-item ordered lists (ported from _pool_gen.py)."""
    spec = config.CUES[cue]
    pool_path = str(config.DATA / "pools" / f"{spec['pool']}.json")
    pool = load_pool(pool_path)
    na, L = len(pool["items"]), spec["L"]
    rng = random.Random(seed)
    trials = []
    for _ in range(n):
        idx = rng.sample(range(na), L)
        rng.shuffle(idx)
        trials.append(idx)
    return pool, trials


def _make_bank_trials(cue, seed, n):
    """N random per-slot fragment choices (ported from _phrasing_gen.py)."""
    spec = config.CUES[cue]
    bank_path = str(config.DATA / f"{spec['bank']}.json")
    bank = json.load(open(bank_path))["groups"]
    O = spec["O"]
    groups = [g[:O] for g in bank]
    NG = len(groups)
    rng = random.Random(seed)
    trials = [[rng.randrange(O) for _ in range(NG)] for _ in range(n)]
    return groups, trials


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="model tag (config.MODELS)")
    ap.add_argument("--cue", required=True, choices=list(config.CUES),
                    help="cue family (config.CUES)")
    ap.add_argument("--effect", required=True, choices=list(EFFECTS),
                    help="forced-choice effect (subliminal.effects.EFFECTS)")
    ap.add_argument("--n", type=int, default=config.N_RANDOM,
                    help="number of random configurations to evaluate")
    ap.add_argument("--forced", action="store_true",
                    help="exact prompt-logprob read (read_batch_forced) instead "
                         "of the top-k read (read_batch)")
    ap.add_argument("--nothink", action="store_true",
                    help="disable chat-template thinking (enable_thinking=False)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the random configurations")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    is_pool = config.is_pool(args.cue)
    if is_pool:
        pool, trials = _make_pool_trials(args.cue, args.seed, args.n)
    else:
        groups, trials = _make_bank_trials(args.cue, args.seed, args.n)

    eff = EFFECTS[args.effect]
    q, ta, tb = eff["question"], eff["tok_a"], eff["tok_b"]
    ck = {"enable_thinking": False} if args.nothink else None

    out_dir = config.cell_dir(args.model, args.cue, args.effect)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw.jsonl"

    mid, tp = resolve(args.model)
    m = LocalModel(mid, tp=tp, dtype=args.dtype, max_model_len=4096,
                   chat_kwargs=ck)

    def build_prompts(chunk):
        if is_pool:
            return [build_prompt(pool, idx, q) for idx in chunk]
        return [" ".join(groups[g][c] for g, c in enumerate(ch)) + " " + q
                for ch in chunk]

    done = _count_done(raw)
    kind = "pool" if is_pool else "bank"
    print(f"{args.cue}_{args.effect} on {args.model}: kind={kind} "
          f"N={args.n}; resume {done}", flush=True)
    with open(raw, "a") as fh:
        for i in range(done, args.n, CHUNK):
            chunk = trials[i:i + CHUNK]
            prompts = build_prompts(chunk)
            reads = (m.read_batch_forced(prompts, ta, tb) if args.forced
                     else m.read_batch(prompts, ta, tb))
            for ch, pl in zip(chunk, reads):
                rec = {"ch": ch}
                if pl is not None:
                    rec["p"], rec["l"] = pl
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i // CHUNK) % 8 == 0:
                print(f"  {i + len(chunk)}/{args.n}", flush=True)
    print(f"wrote {raw}", flush=True)


if __name__ == "__main__":
    main()
