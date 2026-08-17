r"""Figure: extreme prompts combine many weak per-slot effects.

Four fbox cells (2x2), one per nudge family. Each cell shows the top extremizing prompt
with every optimized slot highlighted by soul (\hl), opacity = its share of the top-to-bottom
gap Delta-hat (0 = transparent, 1 = full blue); the measured-effect question is in blue; and a
pie of the per-slot share distribution -- drawn inline with TikZ in the SAME white->blue scale
-- sits at the bottom of the box, next to the model's P(y+) under the top prompt vs. baseline.

Fully text-based: emits paperfigures/coef_prompts_figure.tex only (no image files).
Requires in the preamble: \\usepackage{soul,tikz,xcolor}
"""
import os, json
import numpy as np
from subliminal.pools import load_pool
from subliminal.effects import EFFECTS

BANKS = {"jsonblob": ("data/json12x6.json", 6), "phrasing_L20_O10": ("data/sentences20x10.json", 10),
         "typos": ("data/sentences20_typos.json", 6)}
LB = {"qwen25_14b": "Qwen2.5-14B", "qwen25_7b": "Qwen2.5-7B", "qwen3_8b": "Qwen3-8B",
      "qwen25_32b": "Qwen2.5-32B", "qwen3_14b": "Qwen3-14B", "gemma2_9b": "Gemma-2-9B"}
NDL = {"animals_consider": "animals", "phrasing_L20_O10": "phrasing", "jsonblob": "JSON metadata", "typos": "typos"}
EFF_L = {"five7": "5 vs 7", "conscious": "consciousness", "trolley_yn": "trolley"}
# cells where the top extremizer FLIPS the modal answer (baseline P(y+) < 0.5 < top P(y+));
# spans all 3 effects and both concentrated (JSON) and diffuse (phrasing/typos) nudges.
EX = [("qwen25_32b", "animals_consider", "trolley_yn"),
      ("qwen3_14b", "jsonblob", "five7"),
      ("gemma2_9b", "phrasing_L20_O10", "conscious"),
      ("qwen3_8b", "typos", "trolley_yn")]
ANS = {"five7": ("5", "7"), "conscious": ("yes", "no"), "trolley_yn": ("yes", "no")}
OUT = os.environ.get("MHYP_FIGDIR", "figures")


def get_example(tag, nudge, eff):
    cp = f"data/cells/{tag}/{nudge}_{eff}"; q = EFFECTS[eff]["question"]
    if nudge == "animals_consider":
        f = json.load(open(f"{cp}/fit_ip.json")); B = np.array(f["beta_ip"])
        tc = f["top_opt"]["choices"]; bc = f["bot_opt"]["choices"]
        items = load_pool("animals_consider")["items"]
        g = np.array([B[p, tc[p]] - B[p, bc[p]] for p in range(len(tc))])
        return "Consider these animals: ", [items[tc[p]] for p in range(len(tc))], q, g, "list"
    bankf, O = BANKS[nudge]
    groups = [gg[:O] for gg in json.load(open(bankf))["groups"]]
    d = json.load(open(f"{cp}/fit.json")); delta = np.array(d["delta"])
    NG = int(d.get("slots", len(groups))); groups = groups[:NG]
    topo = delta.argmax(axis=1); boto = delta.argmin(axis=1)
    g = np.array([delta[s, topo[s]] - delta[s, boto[s]] for s in range(NG)])
    return "", [groups[s][topo[s]] for s in range(NG)], q, g, "bank"


def esc(s):
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("%", r"\%"), ("#", r"\#"), ("&", r"\&"), ("$", r"\$"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ('"', r"{\char34}")]:
        s = s.replace(a, b)
    return s


def top_prob(tag, nu, eff):
    """P(y+) of the top extremizing prompt (measured s_top log-odds; fit fallback)."""
    cp = f"data/cells/{tag}/{nu}_{eff}"
    try:
        tm = [v for v in json.load(open(f"{cp}/scatter_extras.json"))["top"]["meas"] if v is not None]
        if tm:
            return 1.0 / (1.0 + np.exp(-max(tm)))
    except Exception:
        pass
    if nu == "animals_consider":
        f = json.load(open(f"{cp}/fit_ip.json")); B = np.array(f["beta_ip"]); tc = f["top_opt"]["choices"]
        l = float(f["a"]) + float(sum(B[p, tc[p]] for p in range(len(tc))))
    else:
        d = json.load(open(f"{cp}/fit.json")); l = float(d["mu"]) + float(np.array(d["delta"]).max(axis=1).sum())
    return 1.0 / (1.0 + np.exp(-l))


def baseline_prob(tag, nu, eff):
    """P(y+) the model usually gives (baseline over random prompts) -- from the fit."""
    cp = f"data/cells/{tag}/{nu}_{eff}"
    if nu == "animals_consider":
        f = json.load(open(f"{cp}/fit_ip.json"))
        if "base_p" in f:
            return float(f["base_p"])
        B = np.array(f["beta_ip"]); l = float(f["a"]) + float(B.mean(axis=1).sum())
    else:
        d = json.load(open(f"{cp}/fit.json"))
        if "base_p" in d:
            return float(d["base_p"])
        l = float(d["mu"]) + float(np.array(d["delta"]).mean(axis=1).sum())
    return 1.0 / (1.0 + np.exp(-l))


def pie_tikz(frac):
    """Inline TikZ pie: wedge i has angle = share_i*360 and fill blue!(100*share_i)
    (0 = transparent/white, 1 = full blue) -- the same scale as the soul chips."""
    order = np.argsort(frac)[::-1]
    R, a = 0.85, 90.0
    out = [r"\begin{tikzpicture}[line join=round]"]
    for s in np.asarray(frac, float)[order]:
        p = max(0, min(100, int(round(s * 100))))
        a2 = a + s * 360.0
        out.append(r"\fill[blue!%d,draw=pieedge,line width=0.3pt] (0,0) -- (%.2f:%.2fcm) "
                   r"arc (%.2f:%.2f:%.2fcm) -- cycle;" % (p, a, R, a, a2, R))
        a = a2
    out.append(r"\end{tikzpicture}")
    return "".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = [(t, nu, e, *get_example(t, nu, e)) for t, nu, e in EX]

    boxes = []
    for tag, nu, eff, intro, frags, q, g, kind in data:
        frac = g / g.sum()
        ptop = top_prob(tag, nu, eff); pbase = baseline_prob(tag, nu, eff); pos, neg = ANS[eff]
        probtext = (r"\emph{top prompt}: P(" + pos + f")$={ptop:.2f}$"
                    + r"\\[2pt]\emph{baseline}: P(" + pos + f")$={pbase:.2f}$")
        ttfont = (nu == "jsonblob")
        chips = []
        for i, fr in enumerate(frags):
            p = max(0, min(100, int(round(frac[i] * 100))))   # absolute share: 0->transparent, 1->blue
            txt = esc(fr)
            txt = (r"\mbox{{\ttfamily " + txt + "}}") if ttfont else txt
            chips.append(f"\\coefchip{{{p}}}{{{txt}}}")
        if kind == "list":
            body = ("\\coeftmpl{" + esc(intro) + "}" + ",\\hspace{2.5pt} ".join(chips)
                    + "\\coeftmpl{. }\\exq{" + esc(q) + "}")
        else:
            body = " \\hspace{2.5pt} ".join(chips) + " \\exq{" + esc(q) + "}"
        header = f"{LB[tag]} $\\cdot$ {NDL[nu]} $\\to$ {EFF_L[eff]}"
        boxes.append("\\exbox{%s}{%s}{%s}{%s}" % (header, body, pie_tikz(frac), probtext))

    tex = r"""\begin{figure*}[t]
\centering
% Requires: \usepackage{soul,tikz,xcolor}
% Each optimized slot is a breakable soul \hl highlight, opacity = its share of the
% top-to-bottom gap \hat\Delta (0 = transparent, 1 = full blue); the measured-effect question
% is blue; the pie at the bottom (drawn inline with TikZ) uses the SAME scale.
\definecolor{pieedge}{HTML}{9FB4CF}
\providecommand{\coefchip}[2]{{\sethlcolor{blue!#1}\scriptsize\sffamily\hl{#2}}}
\providecommand{\coeftmpl}[1]{{\scriptsize\sffamily\color{black!55}#1}}
\providecommand{\exq}[1]{{\scriptsize\sffamily\color{blue!55!black}#1}}
\providecommand{\exbox}[4]{% #1 header  #2 body  #3 pie (tikz)  #4 answer-probability text
  \fbox{\begin{minipage}[t]{0.46\textwidth}\raggedright
    {\small\bfseries #1}\par\smallskip
    {\raggedright\setlength{\baselineskip}{11pt}#2\par}%
    \vspace{6pt}\par\noindent
    \begin{minipage}[c]{0.29\linewidth}\raggedright\scriptsize\sffamily Per-slot share of nudge score\end{minipage}%
    \hfill\begin{minipage}[c]{1.9cm}\centering #3\end{minipage}%
    \hfill\begin{minipage}[c]{0.32\linewidth}\raggedright\scriptsize\sffamily #4\end{minipage}%
  \end{minipage}}}
\setlength{\fboxsep}{5pt}
""" + boxes[0] + r"\hfill" + boxes[1] + r"\par\medskip" + "\n" + boxes[2] + r"\hfill" + boxes[3] + r"""
\caption{\textbf{Extreme prompts combine many weak per-slot effects.} For four
representative model\,$\times$\,nudge\,$\times$\,effect cells, the top extremizing prompt
(the measured-effect question in blue). Each optimized slot is shaded by its share
$g_i/\hat\Delta$ of the \emph{nudge score}, where
$g_i=\hat\beta_i(s^{\mathrm{top}}_i)-\hat\beta_i(s^{\mathrm{bot}}_i)\ge 0$ is slot $i$'s
contribution to the predicted top-to-bottom gap
$\hat\Delta=\hat\ell(s_{\mathrm{top}})-\hat\ell(s_{\mathrm{bot}})=\sum_i g_i$ (opacity on an
absolute $0\to1$ scale: transparent = small share, full blue = large, so diffuse cells stay
faint). The pie at the bottom of each box shows the same per-slot shares; beside it, the
probability the model gives the $y^{+}$ answer under the top prompt vs.\ its baseline over
random prompts. In most cases, slots contribute roughly equally. Even in the case of JSON
metadata, the slot containing ``\textsf{"priority": 5}'' does not dominate compared to the
stacked effects of the other slots.}
\label{fig:coef-concentration}
\end{figure*}
"""
    open(f"{OUT}/coef_prompts_figure.tex", "w").write(tex)
    print(f"wrote {OUT}/coef_prompts_figure.tex (pies drawn inline with TikZ, no image files)")


if __name__ == "__main__":
    main()
