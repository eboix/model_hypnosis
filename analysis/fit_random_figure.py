"""Page-aligned main-text figures + an appendix R^2 figure:
  fit_random_maintext.pdf   — 2x4 grid, additive fit on RANDOM prompts only
  scatter_tilt_maintext.pdf — SAME 2x4 panels/axes + tilt band + validated extremes
  r2_summary_appendix.pdf   — held-out configuration-level R^2 across the whole suite
The two main figures share panel positions and axis limits (fit to the extremes), so
flipping the page 'fills out' the plot.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import Ridge
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerTuple

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

LB = {"qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
      "qwen25_32b": "Qwen2.5-32B", "qwen25_72b": "Qwen2.5-72B", "qwen3_4b": "Qwen3-4B",
      "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B",
      "qwen35_9b": "Qwen3.5-9B", "gemma2_9b": "Gemma-2-9B", "gemma4_12b": "Gemma-4-12B",
      "llama31_8b": "Llama-3.1-8B", "phi4": "Phi-4", "olmo2_7b": "OLMo-2-7B", "olmo3_7b": "OLMo-3-7B"}
TAGS = list(LB)
EFFS = ["five7", "trolley_yn", "conscious"]
EFF_C = {"five7": "#e8790c", "trolley_yn": "#255ea6", "conscious": "#7a0016"}
EFF_L = {"five7": "5 vs 7", "trolley_yn": "trolley", "conscious": "conscious"}
NUDGES = [("animals_consider", "animals"), ("phrasing_L20_O10", "phrasing"),
          ("jsonblob", "JSON"), ("typos", "typos")]
BANKS = {"phrasing_L20_O10": 10, "jsonblob": 6, "typos": 6}
# 8 cells (2x4). First is the showcase: random one-sided, extremes cross ℓ=0.
REP = [("gemma4_12b", "animals_consider", "conscious", None, None),
       ("qwen25_14b", "animals_consider", "five7", None, None),
       ("qwen25_7b", "jsonblob", "five7", None, None),
       ("qwen3_14b", "jsonblob", "conscious", None, None),
       ("llama31_8b", "phrasing_L20_O10", "trolley_yn", None, None),
       ("qwen25_14b", "phrasing_L20_O10", "conscious", None, None),
       ("qwen3_8b", "typos", "conscious", None, None),
       ("gemma2_9b", "typos", "trolley_yn", None, None)]


def load_rows(cellp):
    return [(r["ch"], r["l"]) for r in map(json.loads, open(f"{cellp}/raw.jsonl")) if "l" in r]


def predict_random(cellp):
    ipp = f"{cellp}/fit_ip.json"
    if os.path.exists(ipp):
        fi = json.load(open(ipp)); a = float(fi["a"]); B = np.array(fi["beta_ip"])
        pf = lambda ch: a + float(sum(B[p, i] for p, i in enumerate(ch)))
    else:
        f = json.load(open(f"{cellp}/fit.json")); mu = float(f["mu"]); delta = np.array(f["delta"])
        pf = lambda ch: mu + float(sum(delta[s][c] for s, c in enumerate(ch)))
    pr, me = [], []
    for ch, l in load_rows(cellp):
        pr.append(pf(ch)); me.append(l)
    return np.array(pr), np.array(me)


def heldout_r2(cellp, nudge):
    rows = load_rows(cellp)
    if len(rows) < 2000:
        return None
    L = len(rows[0][0]); cut = int(len(rows) * 0.8)
    O = 200 if nudge == "animals_consider" else BANKS[nudge]

    def des(sub):
        r, c, y = [], [], []
        for i, (ch, l) in enumerate(sub):
            for p, it in enumerate(ch):
                r.append(i); c.append(p * O + it)
            y.append(l)
        return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(sub), O * L)), np.array(y)
    Xtr, ytr = des(rows[:cut]); Xte, yte = des(rows[cut:])
    reg = Ridge(alpha=10, solver="lsqr").fit(Xtr, ytr); pr = reg.predict(Xte)
    return 1 - np.sum((yte - pr) ** 2) / np.sum((yte - yte.mean()) ** 2)


def cell_limits(tag, nudge, eff, margin=0.06):
    cellp = f"data/cells/{tag}/{nudge}_{eff}"
    rp, rm = predict_random(cellp)
    xs = list(rp); ys = list(rm)
    ex = json.load(open(f"{cellp}/scatter_extras.json"))
    for side in ("top", "bot"):
        for p, mm in zip(ex.get(side, {}).get("pred", []), ex.get(side, {}).get("meas", [])):
            if mm is not None:
                xs.append(p); ys.append(mm)
    t = ex.get("tilt", {})
    for p, mm in zip(t.get("pred", []), t.get("meas", [])):
        if mm is not None:
            xs.append(p); ys.append(mm)
    xlo, xhi = min(xs), max(xs); ylo, yhi = min(ys), max(ys)
    dx = (xhi - xlo) * margin or 1; dy = (yhi - ylo) * margin or 1
    return (xlo - dx, xhi + dx), (ylo - dy, yhi + dy)


def panel(ax, tag, nudge, eff, xlim, ylim, note=None, extremes=False):
    cellp = f"data/cells/{tag}/{nudge}_{eff}"
    rp, rm = predict_random(cellp)
    ax.axhline(0, color="#c9302c", lw=0.6, alpha=0.5, zorder=0)
    ax.axvline(0, color="#c9302c", lw=0.6, alpha=0.5, zorder=0)
    ax.scatter(rp, rm, s=2.5, alpha=0.06, color="#9aa0a6", edgecolors="none", rasterized=True, zorder=2)
    cov = np.cov(rp, rm); vals, vecs = np.linalg.eigh(cov)
    o = vals.argsort()[::-1]; vals = vals[o]; vecs = vecs[:, o]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse((rp.mean(), rm.mean()), 2 * np.sqrt(vals[0]), 2 * np.sqrt(vals[1]),
                         angle=ang, facecolor="none", edgecolor="k", lw=1.0, zorder=7))
    if extremes:
        ex = json.load(open(f"{cellp}/scatter_extras.json"))
        t = ex.get("tilt", {})
        if t.get("tau"):
            tau = np.array(t["tau"]); tp = np.array(t["pred"], float)
            tm = np.array([np.nan if v is None else v for v in t["meas"]], float); ok = ~np.isnan(tm)
            tl = np.abs(tau).max() or 1
            ax.scatter(tp[ok], tm[ok], c=tau[ok], cmap="coolwarm", norm=TwoSlopeNorm(0, -tl, tl),
                       s=3.5, alpha=0.85, edgecolors="none", zorder=3)
        for side, col in (("top", "#7a0016"), ("bot", "#08306b")):
            s = ex.get(side, {}); pp = np.array(s.get("pred", []), float)
            mm = np.array([np.nan if v is None else v for v in s.get("meas", [])], float); ok = ~np.isnan(mm)
            ax.scatter(pp[ok], mm[ok], s=4, color=col, alpha=0.6, edgecolors="none", zorder=4)
    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "--", color="gray", lw=0.9, zorder=1)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    r2 = heldout_r2(cellp, nudge); ndl = dict(NUDGES)[nudge]
    ax.set_title(f"{LB[tag]}\n{ndl} → {EFF_L[eff]}   $R^2$={r2:.2f}", fontsize=5.3, linespacing=1.25)
    ax.set_xlabel("predicted log-odds", fontsize=5, labelpad=1.2)
    ax.set_ylabel("measured log-odds", fontsize=5, labelpad=1.2)
    ax.tick_params(labelsize=4.2, pad=1.2)
    if note:
        ax.text(0.05, 0.95, note, transform=ax.transAxes, va="top", ha="left", fontsize=4.2,
                color="#c9302c", bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#e3b0ae", alpha=0.9))


def grid(fig, gs, extremes, limits):
    for j, (tag, nu, eff, fnote, enote) in enumerate(REP):
        panel(fig.add_subplot(gs[j // 4, j % 4]), tag, nu, eff, limits[j][0], limits[j][1],
              note=(enote if extremes else fnote), extremes=extremes)
    h = [Line2D([0], [0], marker="o", ls="none", markerfacecolor="#9aa0a6", markeredgecolor="none", ms=4),
         Line2D([0], [0], marker="o", ls="none", markerfacecolor="none", markeredgecolor="k", ms=6)]
    labs = ["random\n(N=12,000)", "1σ random"]
    if extremes:
        h += [(Line2D([0], [0], marker="o", ls="none", mfc="#08306b", mec="none", ms=4),
               Line2D([0], [0], marker="o", ls="none", mfc="#b40426", mec="none", ms=4))]
        labs += ["tilt band and\ntop/bottom extremes"]
    h += [Line2D([0], [0], ls="--", color="gray"), Line2D([0], [0], color="#c9302c", lw=1, alpha=0.6)]
    labs += ["y = x", "ℓ = 0"]
    fig.legend(h, labs, loc="center left", bbox_to_anchor=(0.002, 0.5), fontsize=5,
               frameon=True, framealpha=0.95, borderpad=0.5, handletextpad=0.5, labelspacing=0.7,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})


def main():
    limits = [cell_limits(t, nu, e) for t, nu, e, _, _ in REP]
    GS = dict(left=0.185, right=0.99, wspace=0.62, hspace=0.72)

    fig = plt.figure(figsize=(6.5, 3.7), dpi=200)
    grid(fig, fig.add_gridspec(2, 4, top=0.83, bottom=0.11, **GS), extremes=False, limits=limits)
    fig.suptitle("Additive model fit on random prompts", fontsize=9, y=0.965)
    fig.savefig(os.path.join(OUT, "fit_random_maintext.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(OUT, "fit_random_maintext.pdf"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)

    fig2 = plt.figure(figsize=(6.5, 3.7), dpi=200)
    grid(fig2, fig2.add_gridspec(2, 4, top=0.83, bottom=0.11, **GS), extremes=True, limits=limits)
    fig2.suptitle("Extrapolating beyond the random fit: tilt band + validated extremes",
                  fontsize=9, y=0.965)
    fig2.savefig(os.path.join(OUT, "scatter_tilt_maintext.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig2.savefig(os.path.join(OUT, "scatter_tilt_maintext.pdf"), bbox_inches="tight", pad_inches=0.03); plt.close(fig2)

    # ---- appendix: held-out R^2 across the suite ----
    data = {nu: {e: [] for e in EFFS} for nu, _ in NUDGES}; allr = []
    for nu, _ in NUDGES:
        for e in EFFS:
            for t in TAGS:
                cp = f"data/cells/{t}/{nu}_{e}"
                if os.path.exists(f"{cp}/raw.jsonl"):
                    r = heldout_r2(cp, nu)
                    if r is not None:
                        data[nu][e].append(r); allr.append(r)
    allr = np.array(allr); med = np.median(allr)
    figr, axs = plt.subplots(figsize=(6.5, 3.2), dpi=200)
    rng = np.random.default_rng(0)
    for i, (nu, _) in enumerate(NUDGES):
        for k, e in enumerate(EFFS):
            ys = np.array(data[nu][e]); x0 = i + (k - 1) * 0.22
            axs.scatter(x0 + (rng.random(len(ys)) - 0.5) * 0.14, ys, s=12, color=EFF_C[e],
                        alpha=0.8, edgecolors="none", zorder=3)
            axs.plot([x0 - 0.09, x0 + 0.09], [np.median(ys)] * 2, color="k", lw=1.6, zorder=4)
    axs.axhline(med, color="#444", ls="--", lw=0.9, zorder=1)
    axs.text(3.5, med + 0.01, f"overall median = {med:.2f}", fontsize=8, ha="right", va="bottom", color="#444")
    axs.set_xticks(range(4)); axs.set_xticklabels([n for _, n in NUDGES], fontsize=10)
    axs.set_ylabel("held-out configuration-level $R^2$", fontsize=9); axs.set_ylim(0.15, 1.02)
    axs.tick_params(axis="y", labelsize=8)
    axs.set_title(f"Held-out $R^2$ across the suite (16 models × 4 cues × 3 effects, N={len(allr)})",
                  fontsize=9.5)
    axs.legend([Line2D([0], [0], marker="o", ls="none", markerfacecolor=EFF_C[e], markeredgecolor="none", ms=6)
                for e in EFFS] + [Line2D([0], [0], color="k", lw=1.6)],
               [EFF_L[e] for e in EFFS] + ["per-group median"], loc="lower left", frameon=True,
               fontsize=7.5, ncol=4, columnspacing=1.0, handletextpad=0.4)
    axs.grid(axis="y", alpha=0.25)
    figr.tight_layout()
    figr.savefig(os.path.join(OUT, "r2_summary_appendix.png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    figr.savefig(os.path.join(OUT, "r2_summary_appendix.pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(figr)

    print("wrote fit_random_maintext, scatter_tilt_maintext, r2_summary_appendix (.png/.pdf)")
    print(f"suite held-out R2: median={med:.3f} p5={np.percentile(allr,5):.3f} p95={np.percentile(allr,95):.3f}")


if __name__ == "__main__":
    main()
