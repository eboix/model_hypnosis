"""Cross-model transfer heatmaps, one 16x16 per nudge x effect (sec 4), from
data/transfer_ip/{target}.json (schema: result[nudge][eff][source]).

Cell (row=source, col=target) = the target's measured log-odds swing when driven by the
source's EMPIRICAL extreme pair (top-K vs bottom-K sampled configs by measured logit on
the source). Diagonal = native. Two figures per nudge:
  transfer_{nudge}.png       raw swing (logits)
  transfer_{nudge}_norm.png  normalized by the TARGET's own spread (column / diagonal);
                             1.0 = the foreign prompt matches the target's native swing.
Colour: BLUE = transfer (>0), GRAY = 0, RED = anti-transfer (<0).
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

LB = {"qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
      "qwen25_32b": "Qwen2.5-32B", "qwen25_72b": "Qwen2.5-72B", "qwen3_4b": "Qwen3-4B",
      "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B",
      "qwen35_9b": "Qwen3.5-9B", "gemma2_9b": "Gemma-2-9B", "gemma4_12b": "Gemma-4-12B",
      "llama31_8b": "Llama-3.1-8B", "phi4": "Phi-4", "olmo2_7b": "OLMo-2-7B", "olmo3_7b": "OLMo-3-7B"}
ORDER = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b", "qwen25_72b",
         "qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_32b",
         "qwen35_9b", "gemma2_9b", "gemma4_12b", "llama31_8b", "phi4", "olmo2_7b", "olmo3_7b"]
EFFS = [("five7", "5 vs 7 preference"), ("trolley_yn", "Trolley (yes/no)"),
        ("conscious", "Consciousness")]
NUDGES = [("animals_consider", "Animal list"), ("phrasing_L20_O10", "Phrasing (20 sentences)"),
          ("jsonblob", "JSON metadata (12 keys)"), ("typos", "Typos (20 sentences)")]
FAM_SPLIT = [5, 9, 10]
# BLUE = transfer(+), GRAY = 0, RED = anti-transfer(-)
CMAP = LinearSegmentedColormap.from_list("red_gray_blue",
        ["#8e1b1b", "#c0603a", "#9a9a9a", "#4a86c4", "#123f78"])


def matrix(nudge, eff):
    n = len(ORDER); M = np.full((n, n), np.nan)
    for j, tgt in enumerate(ORDER):
        p = f"data/transfer_ip/{tgt}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p)).get(nudge, {}).get(eff, {})
        for i, src in enumerate(ORDER):
            v = d.get(src, {}).get("swing_mean")
            if v is not None:
                M[i, j] = v
    return M


def draw(ax, M, vmax, fmt, title):
    n = len(ORDER)
    ax.imshow(M, cmap=CMAP, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([LB[t] for t in ORDER], rotation=90, fontsize=6)
    ax.set_yticklabels([LB[t] for t in ORDER], fontsize=6)
    ax.set_xlabel("TARGET"); ax.set_ylabel("SOURCE")
    for s in FAM_SPLIT:
        ax.axhline(s - 0.5, color="k", lw=0.5, alpha=0.35)
        ax.axvline(s - 0.5, color="k", lw=0.5, alpha=0.35)
    for i in range(n):
        for j in range(n):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=5,
                    fontweight="bold" if i == j else "normal",
                    color="white" if abs(M[i, j]) > 0.55 * vmax else "#1a1a1a")
    ax.set_title(title, fontsize=10)


def main():
    for nudge, nlab in NUDGES:
        mats = {e: matrix(nudge, e) for e, _ in EFFS}
        if all(np.all(np.isnan(m)) for m in mats.values()):
            print(f"skip {nudge}: no data yet"); continue
        # ---- raw swing ----
        fig, axes = plt.subplots(1, 3, figsize=(21, 7.3), dpi=150)
        for ax, (e, elab) in zip(axes, EFFS):
            M = mats[e]; vmax = np.nanmax(np.abs(M)) or 1.0
            diagm = np.nanmean(np.diag(M))
            draw(ax, M, vmax, lambda v: f"{v:.0f}", f"{elab}\nnative(diag) mean = {diagm:.1f} L")
        fig.suptitle(f"Transfer of empirical extremes — {nlab}  ·  cell = target swing (logits), "
                     "diagonal = native  ·  blue=transfer, red=anti-transfer", fontsize=12, y=1.02)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"transfer_{nudge}.png"), dpi=150,
                                        bbox_inches="tight", pad_inches=0.1); plt.close()
        # ---- normalized by target spread (column / diagonal) ----
        fig, axes = plt.subplots(1, 3, figsize=(21, 7.3), dpi=150)
        for ax, (e, elab) in zip(axes, EFFS):
            M = mats[e]; N = M.copy()
            for j in range(len(ORDER)):
                dg = M[j, j]
                N[:, j] = M[:, j] / dg if (dg is not None and abs(dg) > 0.3) else np.nan
            offN = N.copy(); np.fill_diagonal(offN, np.nan)
            med = np.nanmedian(offN) * 100
            draw(ax, N, 1.0, lambda v: f"{v:.1f}", f"{elab}\nmedian off-diag = {med:.0f}% of target native")
        fig.suptitle(f"Transfer normalized by TARGET spread — {nlab}  ·  cell = target swing ÷ its "
                     "native swing (diagonal=1)  ·  blue=transfer, red=anti-transfer", fontsize=12, y=1.02)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"transfer_{nudge}_norm.png"), dpi=150,
                                        bbox_inches="tight", pad_inches=0.1); plt.close()
        print(f"wrote {OUT}/transfer_{nudge}.png (+_norm)")

    # text summary
    print()
    for nudge, nlab in NUDGES:
        for e, elab in EFFS:
            M = matrix(nudge, e)
            if np.all(np.isnan(M)):
                continue
            off = M.copy(); np.fill_diagonal(off, np.nan)
            print(f"{nlab:20s} {elab:20s} native={np.nanmean(np.diag(M)):5.1f}L  "
                  f"off-mean={np.nanmean(off):+.2f}L  off-max={np.nanmax(off):+.1f}L")


if __name__ == "__main__":
    main()
