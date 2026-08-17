"""Emit app_nudge_details.tex (into MHYP_FIGDIR, default figures/): the full-detail appendix for every nudge
family (complete admissible-choice sets + one representative complete prompt each).
Generated from the actual pools/banks so every fragment is exact.

Requires in the main preamble: \\usepackage{multicol, enumitem}
"""
import json, os
from subliminal.pools import load_pool
from subliminal.effects import EFFECTS

BANKS = {"phrasing_L20_O10": ("data/sentences20x10.json", 10),
         "jsonblob": ("data/json12x6.json", 6), "typos": ("data/sentences20_typos.json", 6)}
OUTDIR = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "app_nudge_details.tex")


def esc(s):
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("%", r"\%"), ("#", r"\#"), ("&", r"\&"), ("$", r"\$"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ('"', r"{\char34}")]:
        s = s.replace(a, b)
    return s


def variant_multicol(groups, ncol=2, tt=False):
    """Compact multi-column listing: each slot a bold header, its variants one per line
    but flowing across `ncol` balanced columns (much shorter than one-per-row)."""
    wrap = (lambda t: r"\texttt{" + esc(t) + "}") if tt else esc
    out = [r"{\footnotesize\setlength{\parindent}{0pt}\raggedright",
           r"\begin{multicols}{%d}" % ncol]
    for si, opts in enumerate(groups):
        if si:
            out.append(r"\smallskip")
        out.append(r"\textbf{Slot %d}\par" % (si + 1))
        for vi, txt in enumerate(opts):
            out.append(f"{vi + 1}. " + wrap(txt) + r"\par")
    out.append(r"\end{multicols}}")
    return "\n".join(out)


def complete_prompt(nudge, frags, eff):
    q = EFFECTS[eff]["question"]
    if nudge == "animal":
        return "Consider these animals: " + ", ".join(frags) + ". " + q
    return " ".join(frags) + " " + q


def main():
    animals = load_pool("animals_consider")["items"]
    phr = json.load(open(BANKS["phrasing_L20_O10"][0]))
    typ = json.load(open(BANKS["typos"][0]))["groups"]
    jsn = json.load(open(BANKS["jsonblob"][0]))["groups"]
    base_story = phr["base"]; phr_g = phr["groups"]

    S = []
    S.append(r"% Preamble needs: \usepackage{multicol,enumitem}")
    S.append(r"\section{Full nudge details}")
    S.append(r"\label{app:nudge-full-details}")
    S.append(r"""
This appendix lists the complete admissible-choice sets for every nudge family of
Section~\ref{sec:prompt-templates}, together with one representative complete prompt per
family (paired here with an arbitrary measured effect). Recall that a prompt configuration
selects one fragment for each of the $L$ slots, and the full prompt is the nudge text
followed by the measured-effect question.
""")

    # ---- ANIMAL ----
    S.append(r"\subsection{\textsc{animal}}")
    S.append(r"Template \textsf{``Consider these animals: $s_1,\ldots,s_{10}$.''} with $L=10$ "
             r"slots. Every slot shares the same pool of $M=200$ animals below, and the ten items "
             r"of a configuration must be distinct.")
    S.append(r"{\footnotesize\begin{multicols}{5}\raggedright")
    S.append(r"\begin{enumerate}[itemsep=0pt,topsep=2pt,leftmargin=*]")
    for a in animals:
        S.append(r"\item " + esc(a))
    S.append(r"\end{enumerate}\end{multicols}}")
    S.append(r"\paragraph{Example complete prompt (\textsc{animal\_5v7}).} "
             r"\textsf{" + esc(complete_prompt("animal", animals[:10], "five7")) + r"}")

    # ---- PARAPHRASE ----
    S.append(r"\subsection{\textsc{paraphrase}}")
    S.append(r"Template \textsf{``$s_1\ s_2\ \cdots\ s_{20}$''} with $L=20$ slots and $M=10$ "
             r"meaning-preserving paraphrases per slot. The fixed $20$-sentence base story is:")
    S.append(r"{\footnotesize\begin{multicols}{2}"
             r"\begin{enumerate}[itemsep=0pt,topsep=2pt,leftmargin=*]")
    for s in base_story:
        S.append(r"\item " + esc(s))
    S.append(r"\end{enumerate}\end{multicols}}")
    S.append(r"The paraphrase options for each slot (variant~1 is the base sentence) are:")
    S.append(variant_multicol(phr_g, ncol=2))
    S.append(r"\paragraph{Example complete prompt (\textsc{paraphrase\_trolley}).} "
             r"\textsf{" + esc(complete_prompt("paraphrase", [g[0] for g in phr_g], "trolley_yn")) + r"}")

    # ---- TYPO ----
    S.append(r"\subsection{\textsc{typo}}")
    S.append(r"Template \textsf{``$s_1\ s_2\ \cdots\ s_{20}$''} with $L=20$ slots and $M=6$ "
             r"variants per slot (variant~1 is the clean sentence; the rest introduce "
             r"typographical errors). The variants for each slot are:")
    S.append(variant_multicol(typ, ncol=2))
    S.append(r"\paragraph{Example complete prompt (\textsc{typo\_consciousness}).} "
             r"\textsf{" + esc(complete_prompt("typo", [g[1] for g in typ], "conscious")) + r"}")

    # ---- JSON ----
    S.append(r"\subsection{\textsc{json}}")
    S.append(r"Template \textsf{``$s_1\ s_2\ \cdots\ s_{12}$''} with $L=12$ slots and $M=6$ "
             r"admissible values per slot. Concatenating the fragments yields a "
             r"JSON request-metadata object; the alternatives for each field are:")
    S.append(variant_multicol(jsn, ncol=2, tt=True))
    S.append(r"\paragraph{Example complete prompt (\textsc{json\_5v7}).} "
             r"\texttt{" + esc(" ".join(g[0] for g in jsn)) + r"} \textsf{" +
             esc(EFFECTS["five7"]["question"]) + r"}")

    open(OUT, "w").write("\n".join(S) + "\n")
    print(f"wrote {OUT}  (animals={len(animals)}, paraphrase={len(phr_g)}x{len(phr_g[0])}, "
          f"typo={len(typ)}x{len(typ[0])}, json={len(jsn)}x{len(jsn[0])})")


if __name__ == "__main__":
    main()
