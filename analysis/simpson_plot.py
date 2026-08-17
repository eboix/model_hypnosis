"""Standalone simpson plot: the inverse-Simpson effective number of contributing slots
L_eff = (sum g_i)^2 / sum g_i^2 for every model x nudge x effect cell, grouped by nudge.
Well above 1 everywhere -> the steering is spread across many slots, not one dominant one.

Writes figures/ + paperfigures/simpson_leff.{png,pdf}.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from subliminal.pools import load_pool

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

TAGS = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b", "qwen25_72b", "qwen3_4b",
        "qwen3_8b", "qwen3_14b", "qwen3_32b", "qwen35_9b", "gemma2_9b", "gemma4_12b",
        "llama31_8b", "phi4", "olmo2_7b", "olmo3_7b"]
EFFS = ["five7", "trolley_yn", "conscious"]
NUDGES = [("animals_consider", "animals", "#2a9d3f", 10), ("phrasing_L20_O10", "phrasing", "#255ea6", 20),
          ("jsonblob", "JSON", "#e8790c", 12), ("typos", "typos", "#8e44ad", 20)]


def contributions(tag, nudge, eff):
    cp = f"data/cells/{tag}/{nudge}_{eff}"
    if nudge == "animals_consider":
        ipp = f"{cp}/fit_ip.json"
        if not os.path.exists(ipp):
            return None
        f = json.load(open(ipp)); B = np.array(f["beta_ip"])
        tc = f["top_opt"]["choices"]; bc = f["bot_opt"]["choices"]
        return np.array([B[p, tc[p]] - B[p, bc[p]] for p in range(len(tc))])
    fp = f"{cp}/fit.json"
    if not os.path.exists(fp):
        return None
    f = json.load(open(fp)); delta = np.array(f["delta"])
    return delta.max(axis=1) - delta.min(axis=1)


def neff(g):
    g = np.clip(np.asarray(g), 0, None); s = g.sum()
    return float(s * s / np.sum(g * g)) if s > 0 else float("nan")


def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=150)
    rng = np.random.default_rng(0); allv = []
    for i, (nu, lab, c, L) in enumerate(NUDGES):
        vals = [neff(contributions(t, nu, e)) for t in TAGS for e in EFFS
                if contributions(t, nu, e) is not None and np.sum(contributions(t, nu, e)) > 0]
        vals = np.array(vals); allv += list(vals)
        ax.scatter(i + (rng.random(len(vals)) - 0.5) * 0.42, vals, s=22, color=c, alpha=0.75,
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color="k", lw=2.2, zorder=4)
        ax.plot([i - 0.30, i + 0.30], [L, L], color="#777", lw=1.4, ls=":", zorder=2)
        ax.text(i, L + 0.4, f"$L={L}$", ha="center", va="bottom", fontsize=8, color="#555")
    ax.axhline(1, color="#c0392b", lw=1.3, ls="--", zorder=1)
    ax.text(len(NUDGES) - 0.5, 1.35, "single dominant slot ($L_{\\mathrm{eff}}=1$)",
            fontsize=9, color="#c0392b", ha="right")
    ax.set_xticks(range(len(NUDGES))); ax.set_xticklabels([n for _, n, _, _ in NUDGES], fontsize=11)
    ax.set_ylabel("effective # of contributing slots  $L_{\\mathrm{eff}}$", fontsize=11)
    ax.set_title("Extreme prompts are driven by many weak effects", fontsize=12.5)
    ax.legend(handles=[Line2D([0], [0], color="k", lw=2.2, label="per-cue median"),
                       Line2D([0], [0], color="#777", lw=1.4, ls=":", label="total # slots $L$")],
              loc="upper left", fontsize=8.5, frameon=True)
    ax.grid(axis="y", alpha=0.25); ax.set_ylim(0, max(22, np.max(allv) + 2))
    fig.savefig(os.path.join(OUT, "simpson_leff.png"), dpi=200, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(os.path.join(OUT, "simpson_leff.pdf"), bbox_inches="tight", pad_inches=0.06)
    allv = np.array(allv)
    print(f"wrote figures/ + paperfigures/ simpson_leff.png/.pdf")
    print(f"L_eff across suite: median={np.median(allv):.1f} min={allv.min():.1f} "
          f"max={allv.max():.1f} frac>2={np.mean(allv > 2):.0%}")


if __name__ == "__main__":
    main()
