"""Emit paperfigures/example_prompts_figure.tex: a figure* with one COMPLETE example
prompt per nudge family (nudge text in black, the measured-effect question in colour so
the nudge+effect concatenation is visible). Generated from the actual pools/banks.

Requires in the preamble: \\usepackage{xcolor}
"""
import json, os
from subliminal.pools import load_pool
from subliminal.effects import EFFECTS

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

BANKS = {"phrasing_L20_O10": ("data/sentences20x10.json", 10),
         "jsonblob": ("data/json12x6.json", 6), "typos": ("data/sentences20_typos.json", 6)}
EFF_L = {"five7": "5v7", "trolley_yn": "trolley", "conscious": "consciousness"}
# one representative pairing per nudge (covers all three effects)
EX = [("animals_consider", "animal", "five7"),
      ("jsonblob", "json", "conscious"),
      ("phrasing_L20_O10", "paraphrase", "trolley_yn"),
      ("typos", "typo", "five7")]
OUTFILE = os.path.join(OUT, "example_prompts_figure.tex")
_animals = None


def esc(s):
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("%", r"\%"), ("#", r"\#"), ("&", r"\&"), ("$", r"\$"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ('"', r"{\char34}")]:
        s = s.replace(a, b)
    return s


def parts(nudge, eff):
    """(nudge_text, question_text) so the two can be styled differently."""
    global _animals
    q = EFFECTS[eff]["question"]
    if nudge == "animals_consider":
        if _animals is None:
            _animals = load_pool("animals_consider")["items"]
        return "Consider these animals: " + ", ".join(_animals[:10]) + ".", q
    bankf, O = BANKS[nudge]
    groups = [g[:O] for g in json.load(open(bankf))["groups"]]
    idx = 1 if nudge == "typos" else 0   # typo: variant 1 is perturbed (variant 0 is clean)
    return " ".join(g[min(idx, len(g) - 1)] for g in groups), q


def box(nudge, ndl, eff):
    nud, q = parts(nudge, eff)
    body = esc(nud) + " " + r"\exq{" + esc(q) + "}"
    header = r"\textsc{%s} $\to$ %s" % (ndl, EFF_L[eff])
    return r"\exbox{%s}{%s}" % (header, body)


def main():
    b = [box(nu, ndl, ef) for nu, ndl, ef in EX]
    tex = r"""\begin{figure*}[t]
\centering
% Requires \usepackage{xcolor}. One complete example prompt per nudge family:
% the nudge text is black, the measured-effect question is coloured.
\newcommand{\exq}[1]{\textcolor{blue!55!black}{#1}}
\newcommand{\exbox}[2]{%
  \fbox{\begin{minipage}[t]{0.46\textwidth}\raggedright
    {\sffamily\bfseries\scriptsize #1}\par\smallskip
    {\sffamily\scriptsize\raggedright #2\par}%
  \end{minipage}}}
\setlength{\fboxsep}{4pt}
""" + b[0] + r"\hfill" + b[1] + r"\par\medskip" + "\n" + b[2] + r"\hfill" + b[3] + r"""
\caption{\textbf{Complete example prompts}, one per nudge family (paired with a
representative measured effect). Each prompt concatenates the \textbf{nudge} text (black)
with the \textbf{measured-effect} question (blue). The optimized slots are the animals in
\textsc{animal}, the sentences in \textsc{paraphrase}/\textsc{typo}, and the field values
in \textsc{json}; each slot shows its first admissible fragment, except \textsc{typo}, which
shows a typo-perturbed variant so the errors are visible. Full admissible sets are in
Appendix~\ref{app:nudge-full-details}.}
\label{fig:full-example-prompts}
\end{figure*}
"""
    open(OUTFILE, "w").write(tex)
    print(f"wrote {OUTFILE}")


if __name__ == "__main__":
    main()
