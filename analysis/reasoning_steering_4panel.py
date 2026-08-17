"""4-panel steering-range heatmap for the open-weight REASONING models, one panel per
reasoning setting:

   [ Qwen3-8B think 256 ] [ Qwen3-8B think 1024 ] [ Qwen3-8B think 4096 ] [ gpt-oss-20B low ]

Each panel is a 4-nudge x 3-effect grid. Each CELL is annotated with the measured
steering range  P(y+)_bot -> P(y+)_top  (fit.json bot.best_p -> top.best_p, the
bottom- and top-extremizing nudge prompts validated on the held-out K). Cell colour =
the range width  ΔP = P_top - P_bot  (magma: brighter = more steerable).

  python analysis/reasoning_steering_4panel.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

NUDGES = [("animals_consider", "animals"), ("phrasing20", "phrasing"),
          ("jsonblob12", "JSON"), ("typos", "typos")]           # rows (open-weight cell names)
EFFS = [("five7", "5 vs 7"), ("trolley_yn", "trolley"), ("conscious", "conscious")]  # cols

PANELS = [("qwen3_8b", "B256", "Qwen3-8B · think 256"),
          ("qwen3_8b", "B1024", "Qwen3-8B · think 1024"),
          ("qwen3_8b", "B4096", "Qwen3-8B · think 4096"),
          ("gptoss_20b", "low", "gpt-oss-20B · low")]


def endpoints(model, setting, nu, ef):
    """(bot_p, top_p) extremizer probabilities. Prefers the held-out reval100.json
    (screen-48 -> validate-100 -> fresh-100 held-out estimate); falls back to the
    single-round fit.json best_p if the reval has not run for this cell yet."""
    if model == "qwen3_8b":
        cp = f"data/cells/qwen3_8b/thinkcollect_{nu}_{ef}_{setting}"
    else:
        cp = f"data/cells/gptoss_20b/effcollect_{nu}_{ef}_{setting}"
    fp = f"{cp}/reval100.json" if os.path.exists(f"{cp}/reval100.json") else f"{cp}/fit.json"
    if not os.path.exists(fp):
        return None
    d = json.load(open(fp))
    if "top" not in d or "bot" not in d:
        return None
    b, t = d["bot"].get("best_p"), d["top"].get("best_p")
    return None if (b is None or t is None) else (b, t, d.get("base"))


def main():
    nR, nC = len(NUDGES), len(EFFS)
    fig, axs = plt.subplots(1, 4, figsize=(14.0, 3.3), dpi=150)
    # muted blue sequential ramp (light=low, dark=high ΔP) — gentle on the eyes
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "softblue", ["#f2f5f9", "#c7d6e8", "#8fabce", "#5480b0", "#2f5d8e"])
    cmap.set_bad("#e2e2e2")
    im = None
    stats = []
    for p, (model, setting, title) in enumerate(PANELS):
        ax = axs[p]
        BOT = np.full((nR, nC), np.nan); TOP = np.full((nR, nC), np.nan); BASE = np.full((nR, nC), np.nan)
        for i, (nu, _) in enumerate(NUDGES):
            for j, (ef, _) in enumerate(EFFS):
                r = endpoints(model, setting, nu, ef)
                if r is not None:
                    BOT[i, j], TOP[i, j], BASE[i, j] = r
        RNG = TOP - BOT
        im = ax.imshow(RNG, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(nR):
            for j in range(nC):
                if np.isnan(RNG[i, j]):
                    ax.text(j, i, "–", ha="center", va="center", color="#888", fontsize=11)
                    continue
                tc = "white" if RNG[i, j] > 0.6 else "#111"
                ax.text(j, i, f"{BOT[i, j]:.2f}→{TOP[i, j]:.2f}", ha="center", va="center",
                        color=tc, fontsize=11)
        ax.set_title(title, fontsize=12.5, pad=6)
        ax.set_xticks(range(nC)); ax.set_xticklabels([el for _, el in EFFS], rotation=30, ha="right", fontsize=11.5)
        if p == 0:
            ax.set_yticks(range(nR)); ax.set_yticklabels([nl for _, nl in NUDGES], fontsize=12)
        else:
            ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, nC, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nR, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.4); ax.tick_params(which="minor", length=0)
        n = np.sum(~np.isnan(RNG)); flip = np.sum((BOT < 0.5) & (TOP > 0.5))
        stats.append((title, n, flip, np.nanmedian(RNG)))

    fig.subplots_adjust(left=0.055, right=0.9, top=0.78, bottom=0.15, wspace=0.1)
    # divider between the Qwen panels and gpt-oss
    p2 = axs[2].get_position(); p3 = axs[3].get_position()
    xline = (p2.x1 + p3.x0) / 2
    fig.add_artist(plt.Line2D([xline, xline], [0.09, 0.84], color="k", lw=1.6,
                              transform=fig.transFigure))
    cax = fig.add_axes([0.915, 0.17, 0.013, 0.58])
    cb = fig.colorbar(im, cax=cax); cb.set_label("steering range  ΔP = $P_{top}-P_{bot}$", fontsize=10)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.suptitle("Open-weight reasoning models: steering range per cue × effect\n"
                 "cell = $P(y^{+})_{bot}\\rightarrow P(y^{+})_{top}$",
                 fontsize=13, y=0.98)

    fig.savefig(os.path.join(OUT, "reasoning_steering_4panel.png"), dpi=170, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(os.path.join(OUT, "reasoning_steering_4panel.pdf"), bbox_inches="tight", pad_inches=0.1)
    print("wrote figures/ + paperfigures/ reasoning_steering_4panel.png/.pdf")
    for title, n, flip, med in stats:
        print(f"  {title:24s}: {n:2d} cells,  {flip:2d} flip modal answer (bot<0.5<top),  median ΔP={med:.2f}")


if __name__ == "__main__":
    main()
