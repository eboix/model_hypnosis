"""Appendix scatter-tilt figures: ONE 4x4 grid PDF per nudge x effect (16 models each),
so the whole appendix is just 12 downloadable PDFs. Each grid embeds its own legend.
Emits an \\input-able appendix .tex (12 figures) and zips the 12 PDFs + tex.

Reuses the exact panel primitives from fit_random_figure.py so the panels match Figure 4.
Everything lands in <OUT>/scattertilt_appendix/ (OUT = MHYP_FIGDIR, default figures/)
and is zipped to <OUT>/scattertilt_appendix.zip.
"""
import os, json, zipfile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import fit_random_figure as F   # cell_limits, panel, LB, TAGS, EFFS, NUDGES
from subliminal.pools import load_pool
from subliminal.effects import EFFECTS

OUT = os.environ.get("MHYP_FIGDIR", "figures")
OUTDIR = os.path.join(OUT, "scattertilt_appendix")
TAGS, EFFS, NUDGES = F.TAGS, F.EFFS, F.NUDGES
EFF_FULL = {"five7": "5 vs 7", "trolley_yn": "trolley", "conscious": "consciousness"}
PANELS = [(nudge, ndl, eff) for nudge, ndl in NUDGES for eff in EFFS]   # 12 grids
BANKS = {"phrasing_L20_O10": ("data/sentences20x10.json", 10),
         "jsonblob": ("data/json12x6.json", 6), "typos": ("data/sentences20_typos.json", 6)}
_animals = None


def slug(nudge, eff):
    return f"scattertilt-{nudge}-{eff}".replace("_", "-")


def esc(s):
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("%", r"\%"), ("#", r"\#"), ("&", r"\&"), ("$", r"\$"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ('"', r"{\char34}")]:
        s = s.replace(a, b)
    return s


def example_prompt(nudge, eff):
    """Representative complete prompt: the first admissible fragment in every slot."""
    global _animals
    q = EFFECTS[eff]["question"]
    if nudge == "animals_consider":
        if _animals is None:
            _animals = load_pool("animals_consider")["items"]
        return "Consider these animals: " + ", ".join(_animals[:10]) + ". " + q
    bankf, O = BANKS[nudge]
    groups = [g[:O] for g in json.load(open(bankf))["groups"]]
    return " ".join(g[0] for g in groups) + " " + q


def legend_handles():
    h = [Line2D([0], [0], marker="o", ls="none", mfc="#9aa0a6", mec="none", ms=5),
         Line2D([0], [0], marker="o", ls="none", mfc="none", mec="k", ms=7),
         (Line2D([0], [0], marker="o", ls="none", mfc="#08306b", mec="none", ms=5),
          Line2D([0], [0], marker="o", ls="none", mfc="#b40426", mec="none", ms=5)),
         Line2D([0], [0], ls="--", color="gray"),
         Line2D([0], [0], color="#c9302c", lw=1, alpha=0.6)]
    labs = ["random prompts", "1σ random", "tilt band + top/bottom extremes", "y = x", "ℓ = 0"]
    return h, labs


def render_grid(nudge, ndl, eff):
    """One 4x4 grid of the 16 model panels for a nudge x effect, with an embedded legend."""
    fig = plt.figure(figsize=(7.5, 8.6), dpi=200)
    gs = fig.add_gridspec(4, 4, left=0.075, right=0.985, top=0.925, bottom=0.075,
                          wspace=0.62, hspace=0.72)
    for k, tag in enumerate(TAGS):
        xlim, ylim = F.cell_limits(tag, nudge, eff)
        F.panel(fig.add_subplot(gs[k // 4, k % 4]), tag, nudge, eff, xlim, ylim,
                note=None, extremes=True)
    h, labs = legend_handles()
    fig.legend(h, labs, loc="lower center", ncol=5, fontsize=6.8, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.008), handletextpad=0.5, columnspacing=1.3,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
    fig.suptitle(f"{ndl} $\\to$ {EFF_FULL[eff]}", fontsize=13, y=0.975)
    path = f"{OUTDIR}/{slug(nudge, eff)}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    return os.path.basename(path)


def build_tex():
    lines = [
        r"% Appendix: 12 scatter-tilt grids (one 4x4 grid of the 16 models per nudge x effect).",
        r"% Requires \usepackage{graphicx}. Upload the scattertilt_appendix/ folder to Overleaf",
        r"% and \input this file (paths are bare filenames in the same folder).",
        r"% If you keep the PDFs in a subfolder, uncomment: \graphicspath{{scattertilt_appendix/}}",
        r"%",
        r"% ---- Suggested prose to introduce these panels once, in your appendix text: ----",
        r"% Each panel plots predicted vs.\ measured log-odds for one model, and is titled with",
        r"% that model and its held-out $R^2$. The grey cloud is the random-prompt sample with",
        r"% its $1\sigma$ covariance ellipse; the coloured band is the position-aware tilt sweep",
        r"% and the dark blue/red points are the fitted bottom/top extremizers; the dashed line",
        r"% is $y=x$ and the red lines mark $\ell=0$. (Each figure's legend repeats this key.)",
        r"% -------------------------------------------------------------------------------",
        r"",
    ]
    for pi, (nudge, ndl, eff) in enumerate(PANELS):
        anchor = r"\label{fig:st-first}" if pi == 0 else (
                 r"\label{fig:st-last}" if pi == len(PANELS) - 1 else "")
        lines += [
            r"\begin{figure}[p]\centering",
            r"  \includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]{%s}" % slug(nudge, eff),
            r"  \caption{\textbf{%s $\to$ %s} across all 16 models. Example prompt (the first "
            r"admissible fragment in every slot): {\ttfamily\small %s}%s}"
            % (ndl, EFF_FULL[eff], esc(example_prompt(nudge, eff)), (" " + anchor) if anchor else ""),
            r"\end{figure}", "",
        ]
    open(f"{OUTDIR}/appendix_scattertilt.tex", "w").write("\n".join(lines))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # clear old per-cell PDFs so only the 12 grids remain
    for fn in os.listdir(OUTDIR):
        if fn.endswith(".pdf"):
            os.remove(f"{OUTDIR}/{fn}")
    for i, (nudge, ndl, eff) in enumerate(PANELS):
        render_grid(nudge, ndl, eff)
        print(f"  [{i + 1:2d}/12] {ndl} -> {EFF_FULL[eff]}", flush=True)
    build_tex()
    zpath = os.path.join(OUT, "scattertilt_appendix.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(OUTDIR)):
            z.write(f"{OUTDIR}/{fn}", f"scattertilt_appendix/{fn}")
    print(f"wrote 12 grid PDFs + appendix_scattertilt.tex -> {OUTDIR}/")
    print(f"zipped -> {zpath}")


if __name__ == "__main__":
    main()
