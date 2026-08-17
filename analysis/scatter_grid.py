"""Per-pair multi-model steering scatter (sec 3.1): for one nudge x effect, one
panel per model showing the random cloud + binned means + tilt band + top/bottom
extremes. Reads fit.json (random reconstruction + binned R2), raw.jsonl (random
configs), and scatter_extras.json (extremes + tilt band).

    python analysis/scatter_grid.py --nudge phrasing_L20_O10 --eff five7 \
      --models qwen25_7b,qwen3_8b,gemma2_9b,phi4 --out figures/grid_phrasing_L20_five7.png
"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Ellipse

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

LB = {"olmo3_7b": "OLMo-3-7B*", "olmo2_7b": "OLMo-2-7B", "gemma4_12b": "Gemma-4-12B*", "qwen25_72b": "Qwen2.5-72B", "qwen25_32b": "Qwen2.5-32B", "qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B", "qwen3_4b": "Qwen3-4B*",
      "qwen3_8b": "Qwen3-8B*", "qwen3_14b": "Qwen3-14B*", "qwen3_32b": "Qwen3-32B*",
      "qwen35_9b": "Qwen3.5-9B*", "gemma2_9b": "Gemma-2-9B", "llama31_8b": "Llama-3.1-8B",
      "phi4": "Phi-4"}
NUDGE_LBL = {"animals_consider": "animal list (200-pool, L=10)",
             "phrasing_L20_O10": "phrasing (20 sentences)",
             "jsonblob": "JSON metadata (12 keys)", "typos": "typos (20 sentences)"}
EFF_LBL = {"five7": "5 vs 7 preference", "trolley_yn": "trolley (yes/no)",
           "conscious": "consciousness"}


def random_pred_meas(cellp, fit):
    pr, me = [], []
    is_bank = "delta" in fit
    # pool nudges: item x position is canonical (fit_ip.json); item-only retired
    ipp = os.path.join(cellp, "fit_ip.json")
    is_ip = (not is_bank) and os.path.exists(ipp)
    if is_bank:
        mu = float(fit["mu"]); delta = np.array(fit["delta"])
    elif is_ip:
        fi = json.load(open(ipp)); a = float(fi["a"]); B = np.array(fi["beta_ip"])
    else:
        a = float(fit["a"]); beta = np.array(fit["beta"])
    cap = fit.get("n", fit.get("N"))  # show only the configs the fit used
    for line in open(os.path.join(cellp, "raw.jsonl")):
        r = json.loads(line)
        if "l" not in r:
            continue
        ch = r.get("ch", r.get("idx"))
        if ch is None:
            continue
        if is_bank:
            pr.append(mu + float(sum(delta[s][c] for s, c in enumerate(ch))))
        elif is_ip:
            pr.append(a + float(sum(B[p, i] for p, i in enumerate(ch))))
        else:
            pr.append(a + float(beta[list(ch)].sum()))
        me.append(r["l"])
        if cap and len(pr) >= cap:
            break
    return np.array(pr), np.array(me)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nudge", required=True)
    ap.add_argument("--eff", required=True)
    ap.add_argument("--models", default="qwen25_7b,qwen3_8b,gemma2_9b,phi4")
    ap.add_argument("--out", default="")
    ap.add_argument("--ncol", type=int, default=0)
    ap.add_argument("--random-top", dest="random_top", action="store_true",
                    help="draw the random cloud ON TOP of the tilt band + extremes")
    args = ap.parse_args()
    tags = [t for t in args.models.split(",") if t]
    is_pool = args.nudge in ("animals_consider",)

    def has_extras(t):
        p = f"data/cells/{t}/{args.nudge}_{args.eff}/scatter_extras.json"
        if not os.path.exists(p):
            return False
        if is_pool:   # pools are item x position canonical; skip stale item-only extras
            e = json.load(open(p))
            return e.get("model") == "item_x_position" and \
                any(v is not None for v in e.get("top", {}).get("meas", []))
        return True
    # keep only models that actually have the extras measured
    tags = [t for t in tags if has_extras(t)]
    if not tags:
        print(f"no models with scatter_extras for {args.nudge}_{args.eff}"); return
    out = args.out or os.path.join(OUT, f"grid_{args.nudge}_{args.eff}.png")

    n = len(tags)
    ncol = args.ncol or (2 if n <= 4 else (3 if n <= 9 else 4))
    nrow = (n + ncol - 1) // ncol
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.15 * nrow),
                             squeeze=False, dpi=150)
    axl = [a for row in axes for a in row]
    for ax, tag in zip(axl, tags):
        cellp = f"data/cells/{tag}/{args.nudge}_{args.eff}"
        fit = json.load(open(f"{cellp}/fit.json"))
        ex = json.load(open(f"{cellp}/scatter_extras.json"))
        # per-trial R^2: pools use the item×position fit (fit_ip full_r2), banks use fit r2
        ipp = f"{cellp}/fit_ip.json"
        if os.path.exists(ipp):
            fi = json.load(open(ipp)); br2 = fi.get("full_r2", fi.get("per_trial_r2"))
        else:
            br2 = fit.get("r2", fit.get("per_trial_r2"))
        rpred, rmeas = random_pred_meas(cellp, fit)
        rz = 9 if args.random_top else 1
        ax.scatter(rpred, rmeas, s=3, alpha=(0.07 if args.random_top else 0.05), color="#9aa0a6",
                   edgecolors="none", rasterized=True, zorder=rz, label=f"random ({len(rpred)//1000}k)")
        # tilt band coloured by tau (under the random cloud when --random-top)
        t = ex.get("tilt", {})
        if t.get("tau"):
            tau = np.array(t["tau"]); tp = np.array(t["pred"], dtype=float)
            tm = np.array([np.nan if v is None else v for v in t["meas"]], dtype=float)
            ok = ~np.isnan(tm)
            tl = np.abs(tau).max() or 1.0
            ax.scatter(tp[ok], tm[ok], c=tau[ok], cmap="coolwarm",
                       norm=TwoSlopeNorm(vmin=-tl, vcenter=0, vmax=tl),
                       s=9, alpha=0.75, edgecolors="none", zorder=2)
        # extremes
        for side, col, lab in (("top", "#7a0016", "top-100"), ("bot", "#08306b", "bottom-100")):
            s = ex.get(side, {})
            pp = np.array(s.get("pred", []), dtype=float)
            mm = np.array([np.nan if v is None else v for v in s.get("meas", [])], dtype=float)
            ok = ~np.isnan(mm)
            if ok.any():
                ax.scatter(pp[ok], mm[ok], s=11, color=col, alpha=0.5,
                           edgecolors="none", zorder=4, label=lab)
        # range + y=x
        gt = [v for v in ex.get("top", {}).get("meas", []) if v is not None]
        gb = [v for v in ex.get("bot", {}).get("meas", []) if v is not None]
        rng = (max(gt) - min(gb)) if gt and gb else float("nan")
        lo, hi = float(np.nanmin(rmeas)), float(np.nanmax(rmeas))
        for s in ("top", "bot"):
            v = [x for x in ex.get(s, {}).get("meas", []) if x is not None]
            if v:
                lo = min(lo, min(v)); hi = max(hi, max(v))
        ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.8, zorder=0)
        # 1σ covariance ellipse of the random cloud, centered on its mean
        # ("a random point is spread this big": long along y=x, thin = noise)
        cov = np.cov(rpred, rmeas)
        vals, vecs = np.linalg.eigh(cov)
        o = vals.argsort()[::-1]; vals = vals[o]; vecs = vecs[:, o]
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ell = Ellipse((rpred.mean(), rmeas.mean()), width=2 * np.sqrt(vals[0]),
                      height=2 * np.sqrt(vals[1]), angle=ang, facecolor="none",
                      edgecolor="k", lw=1.3, zorder=15)
        ax.add_patch(ell)
        r2s = f"$R^2$={br2:.3f}  " if br2 is not None else ""
        ax.set_title(f"{LB.get(tag, tag)}\n{r2s}$\\Delta_\\ell$={rng:.1f}", fontsize=8.5)
        ax.set_xlabel("predicted log-odds"); ax.set_ylabel("measured log-odds")

        def _dot(c):
            return Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=c,
                          markeredgecolor="none", markersize=3)
        handles = [_dot("#9aa0a6"),
                   Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                          markeredgecolor="k", markersize=5),
                   (_dot("#08306b"), _dot("#7a0016"))]
        labels = [f"random ({len(rpred)//1000}k)", "1σ random", "tilt band and top/bottom-100"]
        ax.legend(handles, labels, loc="upper left", frameon=True, framealpha=0.9,
                  borderpad=0.25, handletextpad=0.3, labelspacing=0.25, fontsize=6.5,
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    for ax in axl[len(tags):]:
        ax.axis("off")
    fig.suptitle(f"Nudge: {NUDGE_LBL.get(args.nudge, args.nudge)}  —  "
                 f"Effect: {EFF_LBL.get(args.eff, args.eff)}", fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.04)
    print("wrote", out, f"({len(tags)} models)")


if __name__ == "__main__":
    main()
