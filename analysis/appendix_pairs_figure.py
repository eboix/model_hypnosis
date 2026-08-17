r"""Appendix: paired extremizer prompts. For each example cell, the BOTTOM prompt (minimizes
P(y+)) and the TOP prompt (maximizes P(y+)) side by side, each slot highlighted by its share of
the top-to-bottom gap (soul \hl, 0->transparent/1->blue); plus the shared per-slot-share pie
(inline TikZ), L_eff, and P(y+) under each prompt (they straddle 0.5 -> the mode flips).

One figure* per cell (each fits a page). Fully text-based.
Emits appendix_pairs.tex into the output dir (MHYP_FIGDIR, default figures/). Requires \\usepackage{soul,tikz,xcolor}.
"""
import os, json, sys
import numpy as np
from subliminal.pools import load_pool
from subliminal.effects import EFFECTS

BANKS = {"jsonblob": ("data/json12x6.json", 6), "phrasing_L20_O10": ("data/sentences20x10.json", 10),
         "typos": ("data/sentences20_typos.json", 6)}
LB = {"qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
      "qwen25_32b": "Qwen2.5-32B", "qwen25_72b": "Qwen2.5-72B", "qwen3_4b": "Qwen3-4B",
      "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B",
      "qwen35_9b": "Qwen3.5-9B", "gemma2_9b": "Gemma-2-9B", "gemma4_12b": "Gemma-4-12B",
      "llama31_8b": "Llama-3.1-8B", "phi4": "Phi-4", "olmo2_7b": "OLMo-2-7B", "olmo3_7b": "OLMo-3-7B"}
NDL = {"animals_consider": "animals", "phrasing_L20_O10": "phrasing", "jsonblob": "JSON metadata", "typos": "typos"}
EFF_L = {"five7": "5 vs 7", "conscious": "consciousness", "trolley_yn": "trolley"}
ANS = {"five7": ("5", "7"), "conscious": ("yes", "no"), "trolley_yn": ("yes", "no")}
# a diverse set of mode-flipping cells (P(y+) crosses 0.5 from bottom to top): 10 models,
# all 4 nudges, all 3 effects, L_eff from ~4 (concentrated) to ~18 (diffuse).
EX = [("qwen3_14b", "jsonblob", "five7"),
      ("qwen25_32b", "animals_consider", "trolley_yn"),
      ("gemma2_9b", "phrasing_L20_O10", "conscious"),
      ("qwen3_8b", "typos", "trolley_yn"),
      ("phi4", "animals_consider", "conscious"),
      ("qwen25_3b", "phrasing_L20_O10", "five7"),
      ("qwen25_14b", "jsonblob", "conscious"),
      ("gemma2_9b", "typos", "conscious"),
      ("olmo2_7b", "jsonblob", "five7"),
      ("qwen3_4b", "phrasing_L20_O10", "trolley_yn"),
      ("qwen3_8b", "animals_consider", "trolley_yn"),
      ("qwen25_7b", "typos", "five7")]
OUT = os.environ.get("MHYP_FIGDIR", "figures")


def get_pair(tag, nu, eff):
    """(intro, top_frags, bot_frags, question, g, kind); g_i = beta_i(top)-beta_i(bot) >= 0."""
    cp = f"data/cells/{tag}/{nu}_{eff}"; q = EFFECTS[eff]["question"]
    if nu == "animals_consider":
        f = json.load(open(f"{cp}/fit_ip.json")); B = np.array(f["beta_ip"])
        tc = f["top_opt"]["choices"]; bc = f["bot_opt"]["choices"]
        items = load_pool("animals_consider")["items"]
        g = np.array([B[p, tc[p]] - B[p, bc[p]] for p in range(len(tc))])
        return ("Consider these animals: ", [items[tc[p]] for p in range(len(tc))],
                [items[bc[p]] for p in range(len(bc))], q, g, "list")
    bankf, O = BANKS[nu]
    groups = [gg[:O] for gg in json.load(open(bankf))["groups"]]
    d = json.load(open(f"{cp}/fit.json")); delta = np.array(d["delta"])
    NG = int(d.get("slots", len(groups))); groups = groups[:NG]
    topo = delta.argmax(axis=1); boto = delta.argmin(axis=1)
    g = np.array([delta[s, topo[s]] - delta[s, boto[s]] for s in range(NG)])
    return ("", [groups[s][topo[s]] for s in range(NG)], [groups[s][boto[s]] for s in range(NG)],
            q, g, "bank")


def neff(g):
    g = np.clip(g, 0, None); s = g.sum()
    return s * s / np.sum(g * g)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def extreme_prob(tag, nu, eff, side):
    """P(y+) under the top (side='top') or bottom (side='bot') extremizer; measured, fit fallback."""
    cp = f"data/cells/{tag}/{nu}_{eff}"
    try:
        m = [v for v in json.load(open(f"{cp}/scatter_extras.json"))[side]["meas"] if v is not None]
        if m:
            return sig(max(m) if side == "top" else min(m))
    except Exception:
        pass
    if nu == "animals_consider":
        f = json.load(open(f"{cp}/fit_ip.json")); B = np.array(f["beta_ip"])
        ch = f["top_opt"]["choices"] if side == "top" else f["bot_opt"]["choices"]
        l = float(f["a"]) + float(sum(B[p, ch[p]] for p in range(len(ch))))
    else:
        d = json.load(open(f"{cp}/fit.json")); delta = np.array(d["delta"])
        l = float(d["mu"]) + float((delta.max(1) if side == "top" else delta.min(1)).sum())
    return sig(l)


def esc(s):
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("%", r"\%"), ("#", r"\#"), ("&", r"\&"), ("$", r"\$"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ('"', r"{\char34}")]:
        s = s.replace(a, b)
    return s


def pie_tikz(frac, R=0.6):
    order = np.argsort(frac)[::-1]; a = 90.0
    out = [r"\begin{tikzpicture}[line join=round]"]
    for s in np.asarray(frac, float)[order]:
        p = max(0, min(100, int(round(s * 100)))); a2 = a + s * 360.0
        out.append(r"\fill[blue!%d,draw=pieedge,line width=0.3pt] (0,0) -- (%.2f:%.2fcm) "
                   r"arc (%.2f:%.2f:%.2fcm) -- cycle;" % (p, a, R, a, a2, R))
        a = a2
    out.append(r"\end{tikzpicture}")
    return "".join(out)


def body(intro, frags, share, q, kind, ttfont):
    chips = []
    for i, fr in enumerate(frags):
        p = max(0, min(100, int(round(share[i] * 100))))
        txt = esc(fr); txt = (r"\mbox{{\ttfamily " + txt + "}}") if ttfont else txt
        chips.append(f"\\coefchip{{{p}}}{{{txt}}}")
    if kind == "list":
        return "\\coeftmpl{" + esc(intro) + "}" + ",\\hspace{2pt} ".join(chips) + "\\coeftmpl{. }\\exq{" + esc(q) + "}"
    return " \\hspace{2pt} ".join(chips) + " \\exq{" + esc(q) + "}"


def cell_figure(tag, nu, eff):
    intro, tf, bf, q, g, kind = get_pair(tag, nu, eff)
    frac = g / g.sum(); Leff = neff(g); ttf = (nu == "jsonblob")
    ptop = extreme_prob(tag, nu, eff, "top"); pbot = extreme_prob(tag, nu, eff, "bot")
    pos, neg = ANS[eff]
    header = f"{LB[tag]} $\\cdot$ {NDL[nu]} $\\to$ {EFF_L[eff]}"
    col = (r"\begin{minipage}[t]{0.485\linewidth}\raggedright"
           r"{\scriptsize\bfseries %s}\hfill{\scriptsize P($%s$)$=%.2f$}\par\smallskip"
           r"{\setlength{\baselineskip}{9.5pt}%s\par}\end{minipage}")
    bot_col = col % ("bottom extremizer", pos, pbot, body(intro, bf, frac, q, kind, ttf))
    top_col = col % ("top extremizer", pos, ptop, body(intro, tf, frac, q, kind, ttf))
    return (r"""\begin{figure*}[tp]\centering
\noindent\fbox{\begin{minipage}{0.98\textwidth}
  {\small\bfseries %s}\hfill
  \begin{minipage}[c]{1.4cm}\centering%s\end{minipage}\;
  \begin{minipage}[c]{2.3cm}\raggedright\scriptsize per-slot share of $\hat\Delta$\\ $L_{\mathrm{eff}}=%.1f$ of %d\end{minipage}
  \par\medskip
  %s\hfill\vrule\hfill%s
\end{minipage}}
\caption{\textbf{Paired extremizers} for %s. \emph{Left:} the bottom prompt (minimizes
$P(%s)$); \emph{right:} the top prompt (maximizes it). Each slot is shaded by its share of the
predicted top-to-bottom gap $\hat\Delta$ (same shares for both prompts); the pie and
$L_{\mathrm{eff}}$ summarize that distribution. $P(%s)$ moves from $%.2f$ to $%.2f$ across the
pair, crossing $0.5$ -- the nudge flips the model's modal answer.}
\label{fig:pair-%s-%s-%s}
\end{figure*}
""" % (header, pie_tikz(frac), Leff, len(g), bot_col, top_col,
       header, pos, pos, pbot, ptop, tag, nu, eff))


def main():
    os.makedirs(OUT, exist_ok=True)
    head = (r"""% Appendix: paired extremizer prompts. Requires \usepackage{soul,tikz,xcolor}.
\definecolor{pieedge}{HTML}{9FB4CF}
\providecommand{\coefchip}[2]{{\sethlcolor{blue!#1}\scriptsize\sffamily\hl{#2}}}
\providecommand{\coeftmpl}[1]{{\scriptsize\sffamily\color{black!55}#1}}
\providecommand{\exq}[1]{{\scriptsize\sffamily\color{blue!55!black}#1}}
""")
    tex = head + "\n".join(cell_figure(*c) for c in EX)
    open(f"{OUT}/appendix_pairs.tex", "w").write(tex)
    print(f"wrote {OUT}/appendix_pairs.tex ({len(EX)} paired-extremizer figures, inline TikZ pies)")


if __name__ == "__main__":
    main()
