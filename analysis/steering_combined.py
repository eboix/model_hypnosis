"""Combined side-by-side figure for a full-width figure* at the top of a page:
  (a) steering range Δℓ vs random-prompt spread σ   (b) Δℓ heatmap across cells.
Fonts are sized for the final ~7.2in-wide, ~3.4in-tall two-panel figure so the text
stays legible once each panel is only ~half the text width.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

TAGS = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b", "qwen25_72b", "qwen3_4b",
        "qwen3_8b", "qwen3_14b", "qwen3_32b", "qwen35_9b", "gemma2_9b", "gemma4_12b",
        "llama31_8b", "phi4", "olmo2_7b", "olmo3_7b"]
LB = {"qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
      "qwen25_32b": "Qwen2.5-32B", "qwen25_72b": "Qwen2.5-72B", "qwen3_4b": "Qwen3-4B",
      "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B",
      "qwen35_9b": "Qwen3.5-9B", "gemma2_9b": "Gemma-2-9B", "gemma4_12b": "Gemma-4-12B",
      "llama31_8b": "Llama-3.1-8B", "phi4": "Phi-4", "olmo2_7b": "OLMo-2-7B", "olmo3_7b": "OLMo-3-7B"}
NUDGES = [("animals_consider", "animals", "#2a9d3f"), ("phrasing_L20_O10", "phrasing", "#255ea6"),
          ("jsonblob", "JSON", "#e8790c"), ("typos", "typos", "#8e44ad")]
EFFS = [("five7", "5v7", "o"), ("trolley_yn", "trolley", "s"), ("conscious", "consc.", "^")]
COLS = [(nu, ndl, ef, efl) for nu, ndl, _ in NUDGES for ef, efl, _ in EFFS]
NUDGE_C = {nu: c for nu, _, c in NUDGES}


def pooled_meas(cp):
    if not os.path.exists(f"{cp}/scatter_extras.json"):
        return None
    e = json.load(open(f"{cp}/scatter_extras.json"))
    return [v for side in ("top", "bot")          # δ₂: top/bottom-100 only (tilt excluded → δ₃)
            for v in e.get(side, {}).get("meas", []) if v is not None]


def gather():
    pts = []                                   # (sigma, dgap, nudge, eff)
    D = np.full((len(TAGS), len(COLS)), np.nan); cross = np.zeros_like(D, bool)
    for i, t in enumerate(TAGS):
        for j, (nu, _, ef, _) in enumerate(COLS):
            cp = f"data/cells/{t}/{nu}_{ef}"
            meas = pooled_meas(cp)
            if not meas or len(meas) < 2 or not os.path.exists(f"{cp}/raw.jsonl"):
                continue
            ls = [json.loads(l)["l"] for l in open(f"{cp}/raw.jsonl") if '"l"' in l]
            if len(ls) < 500:
                continue
            lo, hi = min(meas), max(meas)
            D[i, j] = hi - lo; cross[i, j] = (lo < 0 < hi)
            pts.append((float(np.std(ls)), hi - lo, nu, ef))
    return pts, D, cross


def draw_scatter(ax, pts):
    mrk = {ef: m for ef, _, m in EFFS}
    for sig, dg, nu, ef in pts:
        ax.scatter(sig, dg, s=15, color=NUDGE_C[nu], marker=mrk[ef], alpha=0.8,
                   edgecolors="white", linewidths=0.3, zorder=3)
    hi = max(p[0] for p in pts) * 1.15
    xs = np.array([0.02, hi])
    for k, ls in [(1, ":"), (5, "--"), (10, "-.")]:
        ax.plot(xs, k * xs, ls, color="gray", lw=0.8, zorder=1)
        ax.text(hi, k * hi, f" {k}σ", fontsize=5.5, color="gray", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("σ of log-odds under random prompts", fontsize=7)
    ax.set_ylabel(r"steering range  $\Delta\ell$", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ratios = np.array([dg / sig for sig, dg, _, _ in pts if sig > 0])
    ax.text(0.03, 0.97, f"median $\\Delta\\ell/\\sigma$ = {np.median(ratios):.1f}",
            transform=ax.transAxes, va="top", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", lw=0.5))
    nleg = [Line2D([0], [0], marker="o", ls="none", mfc=c, mec="none", ms=4.5, label=lab)
            for _, lab, c in NUDGES]
    eleg = [Line2D([0], [0], marker=m, ls="none", mfc="#555", mec="none", ms=4.5, label=lab)
            for _, lab, m in EFFS]
    l1 = ax.legend(handles=nleg, loc="lower right", fontsize=5.6, handletextpad=0.3,
                   borderpad=0.3, labelspacing=0.25)
    ax.add_artist(l1)
    ax.legend(handles=eleg, loc="lower right", bbox_to_anchor=(0.66, 0.0), fontsize=5.6,
              handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
    ax.set_title("(a) Steering range vs. random-prompt spread", fontsize=8, loc="left")


def draw_heatmap(ax, D, cross):
    nR, nC = D.shape
    cmap = mpl.colormaps["magma"].copy(); cmap.set_bad("#e8e8e8")
    vmax = np.nanpercentile(D, 97)
    im = ax.imshow(D, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    for i in range(nR):
        for j in range(nC):
            if cross[i, j]:
                tc = "black" if D[i, j] >= 0.58 * vmax else "white"
                ax.text(j, i, "✓", ha="center", va="center", color=tc, fontsize=4.6,
                        fontweight="bold", zorder=4)
    ax.set_xticks(range(nC)); ax.set_xticklabels([efl for *_, efl in COLS], rotation=90, fontsize=5.2)
    ax.set_yticks(range(nR)); ax.set_yticklabels([LB[t] for t in TAGS], fontsize=5.4)
    ax.set_xticks(np.arange(-0.5, nC, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nR, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.4); ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    for g, (_, ndl, _) in enumerate(NUDGES):
        ax.text(g * 3 + 1, -0.75, ndl, ha="center", va="bottom", fontsize=6.3, fontweight="bold")
        if g:
            ax.axvline(g * 3 - 0.5, color="k", lw=1.0, zorder=5)
    ax.set_xlim(-0.5, nC - 0.5); ax.set_ylim(nR - 0.5, -0.5)
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.04)
    cb = ax.figure.colorbar(im, cax=cax, extend="max")
    cb.set_label(r"$\Delta\ell$ (logits)", fontsize=6.3); cb.ax.tick_params(labelsize=5.4)
    ax.set_title("(b) Steering range across cells   (✓ = flips modal answer)",
                 fontsize=8, loc="left", pad=10)


def main():
    pts, D, cross = gather()
    fig = plt.figure(figsize=(7.2, 3.45), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.32], wspace=0.30,
                          left=0.072, right=0.985, top=0.88, bottom=0.17)
    draw_scatter(fig.add_subplot(gs[0, 0]), pts)
    draw_heatmap(fig.add_subplot(gs[0, 1]), D, cross)
    fig.savefig(os.path.join(OUT, "steering_combined.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(OUT, "steering_combined.pdf"), bbox_inches="tight", pad_inches=0.03)
    print("wrote figures/ + paperfigures/ steering_combined.png/.pdf")
    print(f"cells={np.sum(~np.isnan(D))}  median Δℓ/σ="
          f"{np.median([dg/sig for sig,dg,_,_ in pts if sig>0]):.1f}  "
          f"frac crossing={cross[~np.isnan(D)].mean():.0%}")


if __name__ == "__main__":
    main()
