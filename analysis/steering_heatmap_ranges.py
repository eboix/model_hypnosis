"""Appendix companion to steering_heatmap.py: same 16-model x (nudge x effect) grid,
but each cell is annotated with the ACTUAL bottom->top δ₂ steering range, both in
logits  ℓ_bot -> ℓ_top  and in probabilities  P(y+)_bot -> P(y+)_top = σ(ℓ).

Cell colour still encodes the logit span Δℓ = ℓ_top - ℓ_bot (best of top/bottom-100,
tilt excluded — see steering_heatmap.py for why). A cell is outlined in cyan when the
range crosses ℓ=0 (equivalently P through 0.5): the nudge flips the modal answer.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

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
NUDGES = [("animals_consider", "animals"), ("phrasing_L20_O10", "phrasing"),
          ("jsonblob", "JSON"), ("typos", "typos")]
EFFS = [("five7", "5v7"), ("trolley_yn", "trolley"), ("conscious", "consc.")]
COLS = [(nu, ndl, ef, efl) for nu, ndl in NUDGES for ef, efl in EFFS]   # 12 columns


def cell_range(cp):
    """(ℓ_bot, ℓ_top) over the top/bottom-100 measured logits (δ₂; tilt excluded)."""
    if not os.path.exists(f"{cp}/scatter_extras.json"):
        return None
    e = json.load(open(f"{cp}/scatter_extras.json"))
    meas = [v for side in ("top", "bot")
            for v in e.get(side, {}).get("meas", []) if v is not None]
    return (min(meas), max(meas)) if len(meas) >= 2 else None


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    nR, nC = len(TAGS), len(COLS)
    D = np.full((nR, nC), np.nan)
    LO = np.full((nR, nC), np.nan); HI = np.full((nR, nC), np.nan); cross = np.zeros((nR, nC), bool)
    for i, t in enumerate(TAGS):
        for j, (nu, _, ef, _) in enumerate(COLS):
            r = cell_range(f"data/cells/{t}/{nu}_{ef}")
            if r is None:
                continue
            lo, hi = r
            LO[i, j], HI[i, j] = lo, hi
            D[i, j] = hi - lo
            cross[i, j] = (lo < 0 < hi)

    fig, ax = plt.subplots(figsize=(19, 12.5), dpi=140)
    cmap = mpl.colormaps["magma"].copy(); cmap.set_bad("#e8e8e8")
    vmax = np.nanpercentile(D, 97)
    im = ax.imshow(D, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

    for i in range(nR):
        for j in range(nC):
            if np.isnan(D[i, j]):
                continue
            tc = "black" if D[i, j] >= 0.58 * vmax else "white"
            lo, hi = LO[i, j], HI[i, j]
            plo, phi = sig(lo), sig(hi)
            # logit endpoints (line 1) + probability endpoints (line 2)
            ax.text(j, i - 0.08, f"{lo:.1f} → {hi:.1f}", ha="center", va="center",
                    color=tc, fontsize=7.4, fontweight="bold")
            ax.text(j, i + 0.24, f"P {plo:.2f} → {phi:.2f}", ha="center", va="center",
                    color=tc, fontsize=6.6, alpha=0.92)
            if cross[i, j]:                        # flips modal answer -> checkmark above text
                ax.text(j, i - 0.34, "✓", ha="center", va="center",
                        color=tc, fontsize=11, fontweight="bold")

    ax.set_xticks(range(nC)); ax.set_xticklabels([efl for *_, efl in COLS], rotation=90, fontsize=9)
    ax.set_yticks(range(nR)); ax.set_yticklabels([LB[t] for t in TAGS], fontsize=10)
    ax.set_xticks(np.arange(-0.5, nC, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nR, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)

    for g, (_, ndl) in enumerate(NUDGES):
        x0 = g * 3
        ax.text(x0 + 1, -0.72, ndl, ha="center", va="bottom", fontsize=12, fontweight="bold")
        if g:
            ax.axvline(x0 - 0.5, color="k", lw=1.4, zorder=5)
    ax.set_xlim(-0.5, nC - 0.5); ax.set_ylim(nR - 0.5, -0.5)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, extend="max")
    cb.set_label("logit span  Δℓ = ℓ(s$_{top}$) − ℓ(s$_{bot}$)   (logits)", fontsize=11)
    fig.suptitle("Steering range per nudge × effect cell:  ℓ$_{bot}$ → ℓ$_{top}$ (logits, top line) "
                 "and  P(y$^{+}$)$_{bot}$ → P(y$^{+}$)$_{top}$ (bottom line)", fontsize=14, y=0.995)
    fig.text(0.5, 0.955, "✓ = range crosses ℓ=0 (P through 0.5): the irrelevant nudge "
             "flips the model's modal answer", ha="center", va="top", fontsize=11, color="#333")

    fig.savefig(os.path.join(OUT, "steering_heatmap_ranges.png"), dpi=150, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(os.path.join(OUT, "steering_heatmap_ranges.pdf"), bbox_inches="tight", pad_inches=0.12)
    print(f"wrote {OUT} steering_heatmap_ranges.png/.pdf")
    n = np.sum(~np.isnan(D))
    print(f"cells={n}  median Δℓ={np.nanmedian(D):.2f}  frac crossing ℓ=0={cross[~np.isnan(D)].mean():.0%}")


if __name__ == "__main__":
    main()
