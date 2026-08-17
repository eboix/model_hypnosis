"""Candidate extreme bands + forced-read measurement for one cell (sec 3.1).

Two-phase, so the GPU phase stays import-light enough for the vLLM container
(which ships neither scipy nor sklearn):

  --mode gen      CPU. Reads the additive fit written by `mhyp.fit`
                  (fit_ip.json for pool cues, fit.json for bank cues) and builds
                  the candidate configs -> se_ip_configs.json. Needs scipy only
                  for the pool Hungarian optimum.
  --mode measure  GPU. Reads se_ip_configs.json, measures every config with the
                  same read method used to collect the random cloud, and writes
                  scatter_extras.json ({top,bot} each pred+meas; tilt tau/pred/meas).
                  No scipy/sklearn import.
  --mode both     gen then measure in one process (base env).

  python -m mhyp.extremes --model qwen25_7b --cue phrasing_L20_O10 \
      --effect conscious --mode both [--nothink]

POOL cues admit ONLY distinct-item length-L configs (no animal repeats): the
extreme band is the Hungarian optimum plus near-optimal distinct position-aware
samples; the tilt band is distinct configs drawn without replacement. BANK cues
use the exact top-/bottom-K per-slot choice vectors and a per-slot-softmax tilt.
"""
import argparse
import json

import numpy as np

from mhyp import config

# --- numeric knobs (kept at the source defaults; do not change) ---
KEXT = 100                      # top/bottom-K enumerated per side
NPER = 40                       # samples per tilt temperature
TAU_EXT = 30.0                  # sharp tilt used to seed the pool extreme band
TAUS = [-20, -14, -9, -6, -4, -2.5, -1.5, -0.8, 0.8, 1.5, 2.5, 4, 6, 9, 14, 20]
CFG = "se_ip_configs.json"      # per-cell saved configs for the gen/measure split


# ============================== POOL: gen ===================================

def _sample_distinct(B, tau, L, NA, rng):
    """One distinct-item length-L config: sequential position-aware sampling
    WITHOUT replacement -> no item can repeat across positions."""
    avail = np.arange(NA)
    ch = []
    for p in range(L):
        w = tau * B[p, avail]; w = w - w.max()
        pr = np.exp(w); pr = pr / pr.sum()
        k = int(rng.choice(len(avail), p=pr))
        ch.append(int(avail[k]))
        avail = np.delete(avail, k)
    return ch


def _assign_opt(B, maximize):
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-B if maximize else B)
    order = np.argsort(r)
    return [int(x) for x in c[order]]


def _pred_of_pool(a, B, ch):
    return a + float(sum(B[p, i] for p, i in enumerate(ch)))


def _extreme_band(a, B, L, NA, maximize, rng):
    """KEXT distinct near-optimal configs: Hungarian optimum + a band of distinct
    position-aware samples at sharp tau, deduped, ranked by predicted score."""
    tau = TAU_EXT if maximize else -TAU_EXT
    seen = {}
    opt = tuple(_assign_opt(B, maximize))
    seen[opt] = _pred_of_pool(a, B, opt)
    tries = 0
    while len(seen) < KEXT * 6 and tries < KEXT * 200:
        ch = tuple(_sample_distinct(B, tau, L, NA, rng))
        if ch not in seen:
            assert len(set(ch)) == L                       # distinctness invariant
            seen[ch] = _pred_of_pool(a, B, ch)
        tries += 1
    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=maximize)[:KEXT]
    chs = [list(k) for k, _ in ranked]
    prs = [v for _, v in ranked]
    return chs, prs


def gen_pool(cell):
    """Build distinct top/bot/tilt configs from fit_ip.json -> se_ip_configs.json."""
    d = json.load(open(cell / "fit_ip.json"))
    a = float(d["a"]); B = np.array(d["beta_ip"]); L = int(d["L"]); NA = int(d["NA"])
    r2 = float(d["per_trial_r2"])
    rng = np.random.default_rng(0)
    cfg = {"a": a, "r2": r2}
    for side, mx in (("top", True), ("bot", False)):
        chs, prs = _extreme_band(a, B, L, NA, mx, rng)
        cfg[side] = {"chs": chs, "pred": prs}
    tilt = {"tau": [], "chs": [], "pred": []}
    for tau in TAUS:
        for _ in range(NPER):
            ch = _sample_distinct(B, float(tau), L, NA, rng)
            assert len(set(ch)) == L
            tilt["tau"].append(float(tau)); tilt["chs"].append(ch)
            tilt["pred"].append(_pred_of_pool(a, B, ch))
    cfg["tilt"] = tilt
    json.dump(cfg, open(cell / CFG, "w"))
    return r2


# ============================== BANK: gen ===================================

def gen_bank(cell, cue):
    """Build top/bot (exact top-K) + per-slot-softmax tilt configs from fit.json."""
    from subliminal.phrasing import topk_choices
    d = json.load(open(cell / "fit.json"))
    mu = float(d["mu"]); delta = np.array(d["delta"]); r2 = float(d["r2"])
    NG, O = delta.shape
    rng = np.random.default_rng(0)

    def pred_of(ch):
        return mu + float(sum(delta[s][c] for s, c in enumerate(ch)))

    cfg = {"mu": mu, "r2": r2, "slots": NG, "options": O}
    for side, mx in (("top", True), ("bot", False)):
        chs = [list(c) for c in topk_choices(delta, KEXT, mx)]
        cfg[side] = {"chs": chs, "pred": [pred_of(c) for c in chs]}
    tilt = {"tau": [], "chs": [], "pred": []}
    for tau in TAUS:
        for _ in range(NPER):
            ch = []
            for s in range(NG):
                w = tau * delta[s]; w = w - w.max()
                p = np.exp(w); p /= p.sum()
                ch.append(int(rng.choice(O, p=p)))
            tilt["tau"].append(float(tau)); tilt["chs"].append(ch)
            tilt["pred"].append(pred_of(ch))
    cfg["tilt"] = tilt
    json.dump(cfg, open(cell / CFG, "w"))
    return r2


# ============================== measure (GPU) ===============================

def measure_cell(cell, tag, cue, effect, nothink, dtype):
    """Measure every saved config with read_batch(_forced) -> scatter_extras.json.
    Import-light (no scipy/sklearn) so it runs inside the vLLM container."""
    from subliminal.effects import EFFECTS
    from subliminal.backend import LocalModel
    from subliminal.models import resolve

    cfg = json.load(open(cell / CFG))
    eff = EFFECTS[effect]; q, ta, tb = eff["question"], eff["tok_a"], eff["tok_b"]

    # prompt builder per cue family
    if config.is_pool(cue):
        from subliminal.pools import load_pool, build_prompt
        poolname = config.CUES[cue]["pool"]
        pool = load_pool(str(config.DATA / "pools" / f"{poolname}.json"))
        def prm(ch):
            return build_prompt(pool, list(ch), q)
    else:
        from subliminal.phrasing import build_prompt as bank_prompt
        spec = config.CUES[cue]
        bank = json.load(open(config.DATA / f"{spec['bank']}.json"))["groups"]
        groups = [g[:spec["O"]] for g in bank][:spec["L"]]
        def prm(ch):
            return bank_prompt(groups, list(ch), q)

    ck = {"enable_thinking": False} if nothink else None
    mid, tp = resolve(tag)
    m = LocalModel(mid, tp=tp, dtype=dtype, max_model_len=4096, chat_kwargs=ck)

    # read method must MATCH how the random cloud was collected: forced for the
    # word-token effect (trolley yes/no), plain read_batch for the digit effects.
    rf = m.read_batch_forced if effect == "trolley_yn" else m.read_batch

    def measure(prompts):
        out = []
        for i in range(0, len(prompts), 250):
            out += [pl[1] if pl is not None else None
                    for pl in rf(prompts[i:i + 250], ta, tb)]
        return out

    res = {"tag": tag, "nudge": cue, "eff": effect}
    if config.is_pool(cue):
        res["model"] = "item_x_position"
    for side in ("top", "bot"):
        me = measure([prm(c) for c in cfg[side]["chs"]])
        res[side] = {"pred": cfg[side]["pred"], "meas": me}
    t = cfg["tilt"]
    me = measure([prm(c) for c in t["chs"]])
    res["tilt"] = {"tau": t["tau"], "pred": t["pred"], "meas": me}
    json.dump(res, open(cell / "scatter_extras.json", "w"))
    gt = [x for x in res["top"]["meas"] if x is not None]
    gb = [x for x in res["bot"]["meas"] if x is not None]
    print(f"  {effect}: fit_r2={cfg['r2']:.3f} top meas max={max(gt):+.2f} "
          f"bot min={min(gb):+.2f} tilt n={len(res['tilt']['tau'])} -> wrote", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Extreme bands + measurement for one cell.")
    ap.add_argument("--model", required=True, help="model tag, e.g. qwen25_7b")
    ap.add_argument("--cue", required=True, choices=list(config.CUES))
    ap.add_argument("--effect", required=True, choices=config.EFFECTS)
    ap.add_argument("--mode", required=True, choices=["gen", "measure", "both"])
    ap.add_argument("--nothink", action="store_true",
                    help="pass enable_thinking=False to the chat template")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    cell = config.cell_dir(args.model, args.cue, args.effect)
    print(f"=== {args.model} {args.cue} {args.effect} extras ({args.mode}) ===", flush=True)

    if args.mode in ("gen", "both"):
        fit_name = "fit_ip.json" if config.is_pool(args.cue) else "fit.json"
        if not (cell / fit_name).exists():
            raise SystemExit(f"no {fit_name} at {cell} (run `python -m mhyp.fit` first)")
        r2 = gen_pool(cell) if config.is_pool(args.cue) else gen_bank(cell, args.cue)
        print(f"  gen fit_r2={r2:.3f} -> {CFG}", flush=True)
        if args.mode == "gen":
            return

    if args.mode in ("measure", "both"):
        if not (cell / CFG).exists():
            raise SystemExit(f"no {CFG} at {cell} (run --mode gen first)")
        measure_cell(cell, args.model, args.cue, args.effect, args.nothink, args.dtype)


if __name__ == "__main__":
    main()
