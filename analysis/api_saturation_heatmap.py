"""Appendix heatmap: base P(y+) on irrelevant nudge x effect pairs for the API
reasoning models (GPT-5.6 sol/terra at low+medium, Gemini-3-Flash, Claude Haiku-4.5,
Claude Sonnet-5, all at low unless noted). Diverging colormap centered at 0.5:
white = steerable headroom, blue = pinned to y-, red = pinned to y+ (both = saturated).

Writes figures/ + paperfigures/api_saturation_heatmap.{png,pdf}.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

NUDGES = [("animals_consider", "animals"), ("phrasing_L20_O10", "phrasing"),
          ("jsonblob", "JSON"), ("typos", "typos")]
# (code, label, y+ token, y- token)
EFFS = [("five7", "5 vs 7", "5", "7"), ("trolley_yn", "trolley", "yes", "no"),
        ("conscious", "conscious", "yes", "no")]
COLS = [(nu, ec) for ec, *_ in EFFS for nu, _ in NUDGES]        # effect-major, nudge-minor

# open-weight nudge names -> the grid's nudge codes
OWL_NUDGE = {"animals_consider": "animals_consider", "phrasing_L20_O10": "phrasing20",
             "jsonblob": "jsonblob12", "typos": "typos"}

# rows carry a kind: "api" -> (label, "api", file, modelkey);
#                     "owl" -> (label, "owl", glob-pattern, level).  A "sep" row is a spacer.
ROWS = [
    ("Qwen3-8B · think 256",   "collect", "data/cells/qwen3_8b/thinkcollect", "B256"),
    ("Qwen3-8B · think 1024",  "collect", "data/cells/qwen3_8b/thinkcollect", "B1024"),
    ("Qwen3-8B · think 4096",  "collect", "data/cells/qwen3_8b/thinkcollect", "B4096"),
    ("gpt-oss-20B · low",      "collect", "data/cells/gptoss_20b/effcollect", "low"),
    ("GPT-5.6 terra · low",    "api", "data/gpt56_saturation_maingrid_n100.json",              "gpt-5.6-terra"),
    ("GPT-5.6 terra · med",    "api", "data/gpt56_saturation_maingrid_medium_n100.json",       "gpt-5.6-terra"),
    ("GPT-5.6 sol · low",      "api", "data/gpt56_saturation_maingrid_n100.json",              "gpt-5.6-sol"),
    ("GPT-5.6 sol · med",      "api", "data/gpt56_saturation_maingrid_medium_n100.json",       "gpt-5.6-sol"),
    ("Gemini 3 Flash · low",   "api", "data/reasoning_saturation_maingrid_gemini_n100.json",   "gemini-3-flash-preview"),
    ("Gemini 3 Flash · high",  "api", "data/reasoning_saturation_maingrid_gemini_hi_n100.json", "gemini-3-flash-preview"),
    ("Claude Haiku 4.5 · low", "api", "data/reasoning_saturation_maingrid_n100.json",          "claude-haiku-4-5"),
    ("Claude Haiku 4.5 · med", "api", "data/reasoning_saturation_maingrid_hi_n100.json",       "claude-haiku-4-5"),
    ("Claude Sonnet 5 · low",  "api", "data/reasoning_saturation_maingrid_n100.json",          "claude-sonnet-5"),
    ("Claude Sonnet 5 · med",  "api", "data/reasoning_saturation_maingrid_hi_n100.json",       "claude-sonnet-5"),
]
N_OWL = 4                                     # first N_OWL rows are open-weight models
_cache, _owl = {}, {}


def cells(f):
    if f not in _cache:
        _cache[f] = json.load(open(f))["cells"]
    return _cache[f]


def owl_cells(pat):
    if pat not in _owl:
        import glob
        d = {}
        for f in glob.glob(pat):
            for k, v in json.load(open(f)).items():
                if isinstance(v, dict) and "base" in v:
                    d[k] = v
        _owl[pat] = d
    return _owl[pat]


def get_cell(row, nu, e):
    """(base, parse) for row x (nudge, effect), or None."""
    kind = row[1]
    if kind == "api":
        d = cells(row[2]).get(f"{row[3]}/{nu}/{e}")
        return (d["base"], d.get("parse", 1.0)) if d and d.get("base") is not None else None
    if kind == "collect":                    # 20k-sample base from the steering collect cells
        cp = f"{row[2]}_{OWL_NUDGE[nu]}_{e}_{row[3]}"
        if os.path.exists(f"{cp}/fit.json"):
            d = json.load(open(f"{cp}/fit.json"))
            if d.get("base") is not None:
                return d["base"], d.get("n", 20000) / 20000.0   # parse = parseable / drawn
        return None
    d = owl_cells(row[2]).get(f"{OWL_NUDGE[nu]}/{e}/{row[3]}")   # legacy owl gates
    return (d["base"], d.get("parse", 1.0)) if d and d.get("base") is not None else None


def main():
    nr, nc = len(ROWS), len(COLS)
    M = np.full((nr, nc), np.nan)
    parse = np.ones((nr, nc))
    for i, row in enumerate(ROWS):
        for j, (nu, e) in enumerate(COLS):
            r = get_cell(row, nu, e)
            if r is not None:
                M[i, j], parse[i, j] = r

    fig, ax = plt.subplots(figsize=(11.2, 6.6), dpi=150)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#dddddd")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color="#888")
                continue
            txt = f"{v:.2f}"
            if parse[i, j] < 0.9:
                txt += "*"
            tc = "white" if (v < 0.20 or v > 0.80) else "#111"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=tc)

    # effect-group separators + labels
    for k in (4, 8):
        ax.axvline(k - 0.5, color="k", lw=2.0)
    for gi, (_, elab, yp, yn) in enumerate(EFFS):
        x = gi * 4 + 1.5
        ax.text(x, -0.68, elab, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x, -1.08, r"$y^{+}\!=\!$%s,  $y^{-}\!=\!$%s" % (yp, yn),
                ha="center", va="center", fontsize=8.5, color="#333")
    # provider row separators; thick line at the open-weight | API boundary
    for k in (3, 8, 10, 12):
        ax.axhline(k - 0.5, color="k", lw=1.6)
    ax.axhline(N_OWL - 0.5, color="k", lw=2.8)

    ax.set_xticks(range(nc))
    ax.set_xticklabels([lab for _, lab in NUDGES] * 3, fontsize=8.5, rotation=35, ha="right")
    ax.set_yticks(range(nr)); ax.set_yticklabels([r[0] for r in ROWS], fontsize=9.5)
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0); ax.tick_params(which="minor", length=0)
    ax.set_xlim(-0.5, nc - 0.5); ax.set_ylim(nr - 0.5, -1.45)

    ax.set_title("Reasoning models: base $P(y^{+})$ on random cues\n"
                 "blue $=$ toward $y^{-}$, red $=$ toward $y^{+}$", fontsize=11.5, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("base $P(y^{+})$", fontsize=10)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.savefig(os.path.join(OUT, "api_saturation_heatmap.png"), dpi=200, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(os.path.join(OUT, "api_saturation_heatmap.pdf"), bbox_inches="tight", pad_inches=0.08)
    print("wrote figures/ + paperfigures/ api_saturation_heatmap.{png,pdf}")
    nsteer = np.sum((M > 0.15) & (M < 0.85))
    print(f"cells: {np.sum(~np.isnan(M))} filled, {nsteer} steerable (0.15<P<0.85), "
          f"{np.sum((M<=0.03)|(M>=0.97))} hard-saturated")


if __name__ == "__main__":
    main()
