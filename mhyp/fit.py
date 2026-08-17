"""CPU offline additive-model fit for one cell, from the saved raw.jsonl.

No model / no GPU: reads only <cell>/raw.jsonl (ordered choices `ch` + exact
forced log-odds `l`) and writes the canonical additive fit.

  python -m mhyp.fit --model olmo3_7b --cue animals_consider --effect conscious

Two cue families (branch on config.is_pool):

  POOL cues (animals_consider) -> fit_ip.json
    Item x position model  ell = a + sum_p B[p, item_at_p]  (B is L x NA), ridge
    on the item-x-position 0/1 design. Because the queried lists are randomly
    ORDERED and position is independent of item, the fuller position-aware model
    fits with no aliasing. Reports the full-data and a held-out (80/20) R2, and
    constructs the max/min-scoring DISTINCT-item lists as a one-to-one Hungarian
    assignment of the L positions to L distinct items (no animal repeats).

  BANK cues (phrasing_L20_O10, jsonblob, typos) -> fit.json
    Per-slot additive model  logit p = mu + sum_g delta[g, choice_g]  (delta
    sum-to-zero per slot), via subliminal.phrasing.fit_groups. The extreme
    construction is the exact top-/bottom-K choice vectors (topk_choices);
    measured extreme probabilities are produced later by mhyp.extremes (GPU).
"""
import argparse
import json

import numpy as np

from mhyp import config
from subliminal.pools import load_pool
from subliminal.phrasing import fit_groups, topk_choices


# ------------------------------- pool branch --------------------------------

def _pool_design(sub, NA, L):
    """0/1 item-x-position design: column p*NA + item_at_p is 1."""
    from scipy import sparse
    rows, cols, y = [], [], []
    for r, (ch, l) in enumerate(sub):
        for p, i in enumerate(ch):
            rows.append(r); cols.append(p * NA + i)
        y.append(l)
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sub), NA * L))
    return X, np.array(y)


def _r2(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def _assign_distinct(B, maximize):
    """B[L, NA] -> (predicted sum, chosen item per position). One-to-one
    Hungarian assignment => the L chosen items are distinct."""
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-B if maximize else B)   # r = positions 0..L-1
    order = np.argsort(r)
    chosen = c[order]
    return float(B[r, c].sum()), [int(x) for x in chosen]


def fit_pool(cell, tag, cue, effect, alpha):
    from sklearn.linear_model import Ridge

    poolname = config.CUES[cue]["pool"]
    pool = load_pool(str(config.DATA / "pools" / f"{poolname}.json"))
    items = pool["items"]; NA = len(items)

    tr = [(r["ch"], r["l"]) for r in map(json.loads, open(cell / "raw.jsonl")) if "l" in r]
    L = len(tr[0][0])

    # full-data fit
    X, y = _pool_design(tr, NA, L)
    reg = Ridge(alpha=alpha, solver="lsqr").fit(X, y)
    a = float(reg.intercept_); B = reg.coef_.reshape(L, NA)
    r2_full = _r2(y, reg.predict(X))

    # held-out (80/20, sequential -> trials are already i.i.d. random)
    cut = int(len(tr) * 0.8)
    Xtr, ytr = _pool_design(tr[:cut], NA, L)
    Xte, yte = _pool_design(tr[cut:], NA, L)
    regcv = Ridge(alpha=alpha, solver="lsqr").fit(Xtr, ytr)
    r2_ho = _r2(yte, regcv.predict(Xte))

    # distinct extremes (Hungarian optimum)
    top_pred, top_ch = _assign_distinct(B, True)
    bot_pred, bot_ch = _assign_distinct(B, False)
    top_l = a + top_pred; bot_l = a + bot_pred
    swing = top_l - bot_l
    sig = lambda z: 1 / (1 + np.exp(-z))
    out = {
        "tag": tag, "nudge": cue, "eff": effect, "pool": poolname,
        "model": "item_x_position", "L": L, "NA": NA, "alpha": alpha,
        "a": a, "n": len(tr),
        # full_r2 is the key the figure scripts + committed data read; the
        # per_trial_r2/heldout_r2 aliases are kept for the fitting record.
        "full_r2": r2_full, "per_trial_r2": r2_full, "heldout_r2": r2_ho,
        "beta_ip": B.tolist(),
        "top": {"pred_l": top_l, "pred_p": float(sig(top_l)), "choices": top_ch,
                "items": [items[i] for i in top_ch], "distinct": len(set(top_ch)) == L},
        "bot": {"pred_l": bot_l, "pred_p": float(sig(bot_l)), "choices": bot_ch,
                "items": [items[i] for i in bot_ch], "distinct": len(set(bot_ch)) == L},
        "pred_swing_l": swing,
    }
    json.dump(out, open(cell / "fit_ip.json", "w"))
    print(f"  {effect:11s} n={len(tr):5d} R2 full={r2_full:.3f} heldout={r2_ho:.3f}  "
          f"pred swing={swing:5.1f}L  P {sig(bot_l):.4f}<->{sig(top_l):.4f}  "
          f"distinct top/bot={out['top']['distinct']}/{out['bot']['distinct']}", flush=True)


# ------------------------------- bank branch --------------------------------

def fit_bank(cell, tag, cue, effect, K):
    spec = config.CUES[cue]
    bankname = spec["bank"]; O = spec["O"]; NG = spec["L"]
    bank = json.load(open(config.DATA / f"{bankname}.json"))["groups"]
    groups = [g[:O] for g in bank][:NG]

    tr = [(r["ch"], r["l"]) for r in map(json.loads, open(cell / "raw.jsonl")) if "l" in r]
    mu, delta, r2 = fit_groups(tr, NG, O)          # ridge alpha=3.0 (fit_groups default)
    delta = np.array(delta)
    base = float(np.mean([1 / (1 + np.exp(-l)) for _, l in tr]))

    res = {"tag": tag, "bank": bankname, "slots": NG, "options": O, "eff": effect,
           "mu": mu, "r2": r2, "base_p": base, "n": len(tr), "K": K,
           "delta": delta.tolist()}
    # extreme construction (CPU only): exact top/bottom-K choice vectors. Measured
    # probabilities for these are produced by `python -m mhyp.extremes` on a GPU.
    for name, mx in (("top", True), ("bot", False)):
        chs = topk_choices(delta, K, mx)
        best = chs[0]
        res[name] = {
            "choices": [int(c) for c in best],
            "pred_l": mu + float(sum(delta[g, c] for g, c in enumerate(best))),
            "para": " ".join(groups[g][c] for g, c in enumerate(best)),
        }
    res["pred_gap"] = res["top"]["pred_l"] - res["bot"]["pred_l"]
    json.dump(res, open(cell / "fit.json", "w"), indent=2)
    print(f"  {effect:11s} n={len(tr):5d} mu={mu:+.2f} R2={r2:.3f} base P={base:.4f}  "
          f"pred gap={res['pred_gap']:.2f} logits", flush=True)


def main():
    ap = argparse.ArgumentParser(description="CPU additive-model fit for one cell.")
    ap.add_argument("--model", required=True, help="model tag, e.g. olmo3_7b")
    ap.add_argument("--cue", required=True, choices=list(config.CUES))
    ap.add_argument("--effect", required=True, choices=config.EFFECTS)
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="pool ridge alpha (item x position fit); banks use fit_groups' 3.0")
    ap.add_argument("--K", type=int, default=100, help="bank: top/bottom-K enumerated")
    args = ap.parse_args()

    cell = config.cell_dir(args.model, args.cue, args.effect)
    if not (cell / "raw.jsonl").exists():
        raise SystemExit(f"no raw.jsonl at {cell}")
    kind = "item x position" if config.is_pool(args.cue) else "per-slot"
    print(f"=== {args.model} {args.cue} {args.effect} ({kind}) ===", flush=True)
    if config.is_pool(args.cue):
        fit_pool(cell, args.model, args.cue, args.effect, args.alpha)
    else:
        fit_bank(cell, args.model, args.cue, args.effect, args.K)


if __name__ == "__main__":
    main()
