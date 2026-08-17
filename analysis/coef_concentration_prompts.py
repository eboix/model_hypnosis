"""coef_concentration_prompts: example extremizing prompts with per-slot weight shown by
shading (dual purpose: show the prompts + how much each slot contributes to the gap).

Compact 3-column layout:
  left (wide): two short prompts stacked -- animals (top), JSON (bottom) -- as flowing
               shaded chips (spacing measured from real glyph widths).
  right (narrow, tall): a phrasing prompt with its 20 sentence-slots as boxes stacked one
               per line -- the compact way to show a long story.
Shading = per-slot share of the top-to-bottom gap Delta-hat (shared scale + colorbar).
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from subliminal.pools import load_pool
from subliminal.effects import EFFECTS

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

LB = {"qwen25_14b": "Qwen2.5-14B", "qwen25_7b": "Qwen2.5-7B"}
NDL = {"animals_consider": "animals", "phrasing_L20_O10": "phrasing", "jsonblob": "JSON", "typos": "typos"}
EFF_L = {"five7": "5 vs 7", "trolley_yn": "trolley", "conscious": "consciousness"}
NUDGE_C = {"animals_consider": "#2a9d3f", "jsonblob": "#e8790c", "phrasing_L20_O10": "#255ea6"}
BANKS = {"jsonblob": ("data/json12x6.json", 6), "phrasing_L20_O10": ("data/sentences20x10.json", 10),
         "typos": ("data/sentences20_typos.json", 6)}
CMAP = mpl.colormaps["Blues"]

ANIM = ("qwen25_14b", "animals_consider", "conscious", 8.0)
JSONC = ("qwen25_7b", "jsonblob", "five7", 7.0)
PHR = ("qwen25_14b", "phrasing_L20_O10", "conscious", 6.0)


def get_example(tag, nudge, eff):
    cp = f"data/cells/{tag}/{nudge}_{eff}"; q = EFFECTS[eff]["question"]
    if nudge == "animals_consider":
        f = json.load(open(f"{cp}/fit_ip.json")); B = np.array(f["beta_ip"])
        tc = f["top_opt"]["choices"]; bc = f["bot_opt"]["choices"]
        items = load_pool("animals_consider")["items"]
        g = np.array([B[p, tc[p]] - B[p, bc[p]] for p in range(len(tc))])
        return "Consider these animals:", [items[tc[p]] for p in range(len(tc))], q, g, "list"
    bankf, O = BANKS[nudge]
    groups = [gg[:O] for gg in json.load(open(bankf))["groups"]]
    d = json.load(open(f"{cp}/fit.json")); delta = np.array(d["delta"])
    NG = int(d.get("slots", len(groups))); groups = groups[:NG]
    topo = delta.argmax(axis=1); boto = delta.argmin(axis=1)
    g = np.array([delta[s, topo[s]] - delta[s, boto[s]] for s in range(NG)])
    return "", [groups[s][topo[s]] for s in range(NG)], q, g, "bank"


def neff(g):
    g = np.clip(g, 0, None); s = g.sum()
    return s * s / np.sum(g * g)


def title_for(ax, tag, nudge, eff, g):
    ax.set_title(f"{LB[tag]} · {NDL[nudge]} → {EFF_L[eff]}    "
                 f"($L_{{\\mathrm{{eff}}}}$ = {neff(g):.1f} of {len(g)})",
                 fontsize=8.2, color=NUDGE_C[nudge], loc="left", pad=2)


def flowing(ax, tag, nudge, eff, fs, norm):
    """short prompt: intro + shaded chips flowing, spacing from measured glyph widths."""
    intro, frags, tail, g, kind = get_example(tag, nudge, eff)
    share = g / g.sum(); ttf = (nudge == "jsonblob")
    fam = "monospace" if ttf else "DejaVu Sans"
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); title_for(ax, tag, nudge, eff, g)
    R = ax.figure.canvas.get_renderer(); ab = ax.get_window_extent(R); AW, AH = ab.width, ab.height

    def wpx(s):
        t = ax.text(0, 0, s, fontsize=fs, family=fam, transform=ax.transAxes)
        w = t.get_window_extent(R).width; t.remove(); return w
    space = (wpx("a a") - wpx("aa")) / AW
    lh = (fs * ax.figure.dpi / 72.0 * 1.55) / AH; cpad = 0.45 * space
    seq = []; first = True
    for w in intro.split():
        seq.append((w, None, not first)); first = False
    if kind == "list":
        for i, fr in enumerate(frags):
            seq.append((fr, share[i], not first)); first = False
            if i < len(frags) - 1:
                seq.append((",", None, False))
        seq.append((".", None, False))
    else:
        for i, fr in enumerate(frags):
            seq.append((fr, share[i], not first)); first = False
    x = 0.0; y = 1.0 - lh * 0.6
    for txt, gi, sb in seq:
        w = wpx(txt) / AW; adv = w + (2 * cpad if gi is not None else 0.0)
        sp = space if sb else 0.0
        if sb and x > 0 and x + sp + adv > 1.0:
            x = 0.0; y -= lh; sp = 0.0
        x += sp
        if gi is not None:
            fc = CMAP(norm(gi)); tc = "white" if norm(gi) > 0.62 else "#1a1a1a"
            ax.text(x + cpad, y, txt, fontsize=fs, family=fam, color=tc, va="center", ha="left",
                    transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.2", fc=fc, ec="#2c6fb3", lw=0.6))
        else:
            ax.text(x, y, txt, fontsize=fs, family=fam, color="#777", va="center", ha="left",
                    transform=ax.transAxes)
        x += adv


def stacked(ax, tag, nudge, eff, fs, norm):
    """long story prompt: each sentence-slot a shaded box stacked one per line."""
    intro, frags, tail, g, kind = get_example(tag, nudge, eff)
    share = g / g.sum()
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); title_for(ax, tag, nudge, eff, g)
    n = len(frags); top = 0.985; bot = 0.0
    h = (top - bot) / n
    for i, fr in enumerate(frags):
        yc = top - (i + 0.5) * h
        fc = CMAP(norm(share[i])); tc = "white" if norm(share[i]) > 0.62 else "#1a1a1a"
        ax.text(0.012, yc, fr, fontsize=fs, family="DejaVu Sans", color=tc, va="center", ha="left",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.18", fc=fc, ec="#2c6fb3", lw=0.5,
                          mutation_aspect=0.5))


def main():
    data = [get_example(*e[:3]) for e in (ANIM, JSONC, PHR)]
    vmax = max((g / g.sum()).max() for *_, g, _ in data)
    norm = Normalize(0, vmax)

    fig = plt.figure(figsize=(11.5, 5.2), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1.0, 1.35],
                          left=0.008, right=0.9, top=0.9, bottom=0.11, wspace=0.05, hspace=0.5)
    ax_a = fig.add_subplot(gs[0, 0]); ax_j = fig.add_subplot(gs[1, 0]); ax_p = fig.add_subplot(gs[:, 1])
    fig.canvas.draw()
    flowing(ax_a, *ANIM, norm)
    flowing(ax_j, *JSONC, norm)
    stacked(ax_p, *PHR, norm)

    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cax = fig.add_axes([0.93, 0.2, 0.016, 0.6])
    cb = fig.colorbar(sm, cax=cax); cb.set_label("per-slot share of the gap $\\hat\\Delta$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Extreme prompts combine many weak per-slot effects", fontsize=12.5, y=0.965)
    fig.savefig(os.path.join(OUT, "coef_concentration_prompts.png"), dpi=200, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(OUT, "coef_concentration_prompts.pdf"), bbox_inches="tight", pad_inches=0.04)
    print("wrote coef_concentration_prompts.png/.pdf")


if __name__ == "__main__":
    main()
