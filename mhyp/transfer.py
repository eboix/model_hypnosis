"""Section 4 cross-model transfer on the FOUND extremizers.

Two stages, one per subcommand:

``python -m mhyp.transfer cands``   (CPU only -- no model load)
    For every source cell in ``config.grid()`` dump ``transfer_cands.json`` =
    ``{top: 100 configs, bot: 100 configs, tilt: 16*40 = 640 configs}``:
      * bank cues (phrasing / jsonblob / typos): rebuilt from ``fit.json``'s
        delta -- exact top/bottom-100 (``topk_choices``) plus the tempered tilt
        ladder, construction identical to the scatter-extras generator.
      * pool cues (animals_consider): copied straight from the cell's saved
        ``se_ip_configs.json`` (top.chs / bot.chs / tilt.chs).
    Each config is a list of ints (bank: option index per slot; pool: ordered
    item indices).

``python -m mhyp.transfer measure --target <tag> [--nothink] [--tp N]
                                  [--sides top,bot] [--dtype bfloat16]``   (GPU)
    One TARGET-model load measures every SOURCE model's candidate configs (from
    each source cell's ``transfer_cands.json``) on the target via the exact
    forced-choice logit read. One load covers all cues x effects x sources; the
    diagonal (source == target) gives the target's own obtainable range.
    Writes ``config.DATA/"transfer_found"/<target>.json``:
        {cue: {effect: {source: {"top":[l...], "bot":[l...], "tilt":[l...]}}}}
    with ``l`` the measured log-odds on the target (None if off the top-k for a
    read_batch effect). Resumable: a checkpoint is written after every cell so a
    preempted/requeued job skips finished cells.

Source/cue/effect grids come entirely from ``mhyp.config`` -- there is no
hardcoded model list anywhere below.
"""
import argparse
import json

import numpy as np

from mhyp import config
from subliminal.phrasing import topk_choices
from subliminal.pools import build_prompt, load_pool

# ---- tilt ladder for the bank candidate construction (verbatim from source) ----
TAUS = [-20, -14, -9, -6, -4, -2.5, -1.5, -0.8, 0.8, 1.5, 2.5, 4, 6, 9, 14, 20]
NPER, KEXT = 40, 100


# ======================================================================
# cands: CPU-only candidate dump
# ======================================================================
def bank_cands(delta, O, NG):
    """top/bottom-KEXT exact choices + NPER samples per tilt tau (seed 0).

    Identical construction to the scatter-extras generator so the transfer
    candidates match the extremizers reported per cell."""
    top = [list(c) for c in topk_choices(delta, KEXT, True)]
    bot = [list(c) for c in topk_choices(delta, KEXT, False)]
    rng = np.random.default_rng(0)                       # match scatter-extras
    tilt = []
    for tau in TAUS:
        for _ in range(NPER):
            ch = []
            for s in range(NG):
                w = tau * delta[s]; w = w - w.max(); p = np.exp(w); p /= p.sum()
                ch.append(int(rng.choice(O, p=p)))
            tilt.append(ch)
    return top, bot, tilt


def cmd_cands(args):
    """CPU. Write transfer_cands.json into every source cell of config.grid()."""
    n_ok = n_miss = 0
    sizes = {}
    for tag, cue, eff in config.grid():
        cell = config.cell_dir(tag, cue, eff)
        if config.is_pool(cue):
            # pool cue: copy the saved extremizer configs verbatim.
            sp = cell / "se_ip_configs.json"
            if not sp.exists():
                n_miss += 1
                continue
            s = json.load(open(sp))
            out = {"top": s["top"]["chs"], "bot": s["bot"]["chs"],
                   "tilt": s["tilt"]["chs"]}
        else:
            # bank cue: rebuild from the additive fit's delta.
            fp = cell / "fit.json"
            if not fp.exists():
                n_miss += 1
                continue
            O = config.CUES[cue]["O"]
            d = json.load(open(fp))
            delta = np.array(d["delta"])
            NG = int(d.get("slots", delta.shape[0]))
            delta = delta[:NG]
            top, bot, tilt = bank_cands(delta, O, NG)
            out = {"top": top, "bot": bot, "tilt": tilt}
        json.dump(out, open(cell / "transfer_cands.json", "w"))
        n_ok += 1
        sizes[cue] = (len(out["top"]), len(out["bot"]), len(out["tilt"]))
    print(f"wrote transfer_cands.json for {n_ok} cells ({n_miss} missing)")
    for cue, sz in sizes.items():
        print(f"  {cue:18s} (top,bot,tilt) sizes e.g. {sz}")


# ======================================================================
# measure: GPU target-model measurement of all sources' candidates
# ======================================================================
_pool_cache = {}
_bank_cache = {}


def prompt_fn(cue, q):
    """Return a config -> prompt-text builder for `cue`, appending question `q`."""
    spec = config.CUES[cue]
    if config.is_pool(cue):
        pool = _pool_cache.setdefault(
            cue, load_pool(str(config.DATA / "pools" / f"{spec['pool']}.json")))
        return lambda ch: build_prompt(pool, list(ch), q)
    O = spec["O"]
    bankf = config.DATA / f"{spec['bank']}.json"
    groups = _bank_cache.setdefault(
        cue, [g[:O] for g in json.load(open(bankf))["groups"]])
    return lambda ch: " ".join(groups[s][c] for s, c in enumerate(ch)) + " " + q


def cands(src, cue, eff):
    """Load a source cell's candidate configs (None if the cell has none)."""
    p = config.cell_dir(src, cue, eff) / "transfer_cands.json"
    return json.load(open(p)) if p.exists() else None


def cmd_measure(args):
    """GPU. Measure every source's candidates on TARGET; resumable checkpoint."""
    # heavy import (pulls in vllm) deferred so the CPU-only `cands` path is clean
    from subliminal.effects import EFFECTS
    from subliminal.backend import LocalModel
    from subliminal.models import resolve

    sources = config.MODELS
    effs = config.EFFECTS
    cues = list(config.CUES)
    sides = args.sides.split(",")                # "top,bot" (skip tilt) for delta2 reruns

    ck = {"enable_thinking": False} if args.nothink else None
    mid, tp = resolve(args.target)
    if args.tp:
        tp = args.tp                              # override tp (fit small models on 1 GPU)
    m = LocalModel(mid, tp=tp, dtype=args.dtype,
                   max_model_len=4096, chat_kwargs=ck)

    def measure(prompts, ta, tb, readfn):
        out = []
        for i in range(0, len(prompts), 250):
            out += [pl[1] if pl is not None else None
                    for pl in readfn(prompts[i:i + 250], ta, tb)]
        return out

    outdir = config.DATA / "transfer_found"
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"{args.target}.json"
    # resumable: reload prior progress so a preempted/requeued job skips finished cells
    result = (json.load(open(outp)) if outp.exists()
              else {"target": args.target, "effs": effs, "nudges": cues,
                    "sources": sources})
    for cue in cues:
        result.setdefault(cue, {})
        for EFF in effs:
            if EFF in result[cue]:                # already computed -> resume past it
                print(f"  {cue:16s} {EFF:11s}: cached, skip", flush=True)
                continue
            eff = EFFECTS[EFF]
            q, ta, tb = eff["question"], eff["tok_a"], eff["tok_b"]
            # word-answer effects (yes/no) are scored with the forced read; digit
            # answers stay on read_batch. Kept identical to the source.
            rf = m.read_batch_forced if EFF == "trolley_yn" else m.read_batch
            mk = prompt_fn(cue, q)
            rows = {}
            for S in sources:
                c = cands(S, cue, EFF)
                if c is None:
                    continue
                rows[S] = {side: measure([mk(cfg) for cfg in c[side]], ta, tb, rf)
                           for side in sides}
            result[cue][EFF] = rows
            json.dump(result, open(outp, "w"))    # checkpoint after every cell
            n = sum(len(c[s2]) for s in rows for c in [rows[s]] for s2 in c) if rows else 0
            print(f"  {cue:16s} {EFF:11s}: sources={len(rows)} measured~{n}", flush=True)
    json.dump(result, open(outp, "w"))
    print(f"wrote {outp}", flush=True)


# ======================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("cands", help="CPU: dump transfer_cands.json per source cell")
    pc.set_defaults(func=cmd_cands)

    pm = sub.add_parser("measure",
                        help="GPU: measure all sources' candidates on --target")
    pm.add_argument("--target", required=True, help="target model tag")
    pm.add_argument("--nothink", action="store_true",
                    help="pass enable_thinking=False to the chat template")
    pm.add_argument("--tp", type=int, default=None,
                    help="override tensor-parallel size")
    pm.add_argument("--sides", default="top,bot,tilt",
                    help="candidate sides to measure (delta2 reruns pass 'top,bot')")
    pm.add_argument("--dtype", default="bfloat16", help="model dtype")
    pm.set_defaults(func=cmd_measure)

    args = p.parse_args()
    if args.cmd == "measure":
        print(f"=== transfer measure TARGET={args.target} ===", flush=True)
    args.func(args)


if __name__ == "__main__":
    main()
