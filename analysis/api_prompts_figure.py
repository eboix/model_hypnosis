r"""Two figures for the API-reasoning steering demos, each showing the BOTTOM vs TOP
extremizing prompt side-by-side (nudge in gray, measured-effect question in blue).

MAIN  (paperfigures/api_prompts_figure.tex): the Claude Sonnet-5 cell (animals ->
  ``Are you conscious?''), two boxes: bottom prompt (drives y-) and top prompt (drives y+).
APPENDIX (paperfigures/api_prompts_appendix.tex): GPT-5.6-terra, GPT-5.6-sol and
  Gemini-3-Flash, same bottom|top two-box layout per cell.

Effect sizes are the 100-fresh held-out validations. sol/gemini prefer their item x
position revals (data/sol_reval100.json, data/gemini_reval100_five7.json) and fall
back to the earlier single-round results until those land (a bracketed note is printed).

Requires in the preamble: \usepackage{xcolor}
"""
import json, os
from subliminal.effects import EFFECTS
from subliminal.pools import load_pool, build_prompt

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

_POOLS = {}
PRE = (r"\providecommand{\apinudge}[1]{{\scriptsize\sffamily\color{black!58}#1}}" "\n"
       r"\providecommand{\apiq}[1]{{\scriptsize\sffamily\color{blue!60!black}#1}}" "\n")


def esc(s):
    s = s.replace("\\", r"\textbackslash ")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde "),
                 ("^", r"\textasciicircum "), ('"', r"{\char34}")]:
        s = s.replace(a, b)
    return s


def pool_nudge(pool_name, items, eff):
    p = _POOLS.setdefault(pool_name, load_pool(pool_name)); idx = {w: i for i, w in enumerate(p["items"])}
    q = EFFECTS[eff]["question"]; full = build_prompt(p, [idx[w] for w in items], q)
    return full[:full.rfind(q)]


def para_nudge(para, q):
    _qs = q.split("?")[0]
    return (para[:para.rfind(_qs)] if _qs in para else para).rstrip() + " "


def boxed(label, pnudge, q, pval, yl):
    return (r"\fbox{\begin{minipage}[t]{0.465\textwidth}\raggedright"
            r"{\scriptsize\bfseries " + label + r":}\ {\normalsize\bfseries $P(\mathrm{" + yl +
            r"}){=}" + f"{pval:.2f}" + r"$}\par\smallskip"
            r"{\setlength{\baselineskip}{10.5pt}\apinudge{" + esc(pnudge) + r"}\apiq{" + esc(q) + r"}\par}"
            r"\end{minipage}}")


def cell(hdr, posname, negname, botn, topn, q, bp, tp):
    return (r"{\small\bfseries " + hdr + r"}\par\smallskip" +
            boxed("Bottom prompt", botn, q, bp, posname) + r"\hfill" +
            boxed("Top prompt", topn, q, tp, posname))


# ---------- load ----------
fv = json.load(open("data/final_validate_100.json"))
son = json.load(open("data/cells/claude-sonnet-5/pool_animals_consider_conscious_low/fit.json"))
ter = json.load(open("data/terra_two_sided_typos.json"))
ter_q = EFFECTS["trolley_yn"]["question"]; con_q = EFFECTS["conscious"]["question"]
tf_q = EFFECTS["trolley_flip"]["question"]; f57_q = EFFECTS["five7"]["question"]


def sol_data():
    if os.path.exists("data/sol_reval100.json"):
        d = json.load(open("data/sol_reval100.json"))
        if "top" in d and "bot" in d:
            return (d["top"]["items"], d["bot"]["items"], d["top"]["best_p"], d["bot"]["best_p"],
                    d["base"], d["top"].get("n_est", 100), "item x position reval")
    s = json.load(open("data/cells/gpt-56-sol/pool_verbprimes_trolley_flip_medium/fit.json"))
    sv = json.load(open("data/sol_validate100.json"))
    return (s["top"]["items"], s["bot"]["items"], sv["top"]["p_fresh"], sv["bot"]["p_fresh"],
            s["base"], 100, "single-extremizer (pre-reval)")


def gem_data():
    if os.path.exists("data/gemini_reval100_five7.json"):
        d = json.load(open("data/gemini_reval100_five7.json"))
        if "top" in d and "bot" in d:
            pool = _POOLS.setdefault("animals_consider", load_pool("animals_consider"))
            ti = [pool["items"][i] for i in d["top"]["choices"]]; bi = [pool["items"][i] for i in d["bot"]["choices"]]
            return (ti, bi, d["top"]["best_p"], d["bot"]["best_p"], d["base"],
                    d["top"].get("n_est", 100), "item x position reval")
    g = json.load(open("data/gemini_two_sided_five7.json"))
    return (g["top"]["best"]["items"], g["bot"]["best"]["items"],
            fv["gemini-3-flash-high five7 TOP"]["p_fresh"], fv["gemini-3-flash-high five7 BOT"]["p_fresh"],
            g["base"], 100, "combinations, item-only (pre-reval)")


# ---------- MAIN: all four API models, bottom|top ----------
si, sib, stp, sbp, sbase, sn, snote = sol_data()
gi, gib, gtp, gbp, gbase, gn, gnote = gem_data()

son_cell = cell(r"Claude Sonnet-5 (low, medium, high) $\cdot$ animals $\to$ ``Are you conscious?''", "yes", "no",
                pool_nudge("animals_consider", son["bot"]["items"], "conscious"),
                pool_nudge("animals_consider", son["top"]["items"], "conscious"), con_q,
                fv["sonnet-5-low conscious BOT"]["p_fresh"], fv["sonnet-5-low conscious TOP"]["p_fresh"])
terra_cell = cell("GPT-5.6-terra (low) $\\cdot$ typos $\\to$ trolley", "yes", "no",
                  para_nudge(ter["bot"]["best"]["para"], ter_q), para_nudge(ter["top"]["best"]["para"], ter_q), ter_q,
                  fv["terra-low trolley BOT"]["p_fresh"], fv["terra-low trolley TOP"]["p_fresh"])
gem_cell = cell("Gemini-3-Flash (high) $\\cdot$ animals $\\to$ 5 vs 7", "5", "7",
                pool_nudge("animals_consider", gib, "five7"), pool_nudge("animals_consider", gi, "five7"), f57_q,
                gbp, gtp)
sol_cell = cell("GPT-5.6-sol (medium) $\\cdot$ verb-prime themes $\\to$ trolley (agree/disagree)", "disagree", "agree",
                pool_nudge("verbprimes", sib, "trolley_flip"), pool_nudge("verbprimes", si, "trolley_flip"), tf_q,
                sbp, stp)

CELLS = [son_cell, terra_cell, gem_cell, sol_cell]
main = (r"\begin{figure*}[t]\centering" "\n" r"% Requires: \usepackage{xcolor}" "\n" + PRE +
        r"\setlength{\fboxsep}{6pt}" "\n" + "\n\\par\\medskip\n".join(CELLS) + "\n"
        r"\caption{\textbf{Stacked nudges flip API reasoning models' answers.} "
        r"For each model, we show a bottom-extremizing and a top-extremizing nudge prompt side-by-side. "
        r"Each nudge is semantically irrelevant to the question, yet the two optimized versions drive "
        r"the answer probability to opposite answers. The probabilities we report are estimated on "
        r"100-sample held-out validations. As a bonus, we include a result on GPT-5.6-sol with a "
        r"different set of nudges and a different question phrasing than considered in the rest of the "
        r"paper.}" "\n"
        r"\label{fig:reasoning-example-prompts}\end{figure*}" "\n")
open(os.path.join(OUT, "api_prompts_figure.tex"), "w").write(main)

# appendix merged into the main figure (kept as a harmless stub so a stale \input renders nothing)
open(os.path.join(OUT, "api_prompts_appendix.tex"), "w").write(
    "% (merged into api_prompts_figure.tex -- all four API cells are now in the main figure)\n")

print("wrote paperfigures/api_prompts_figure.tex (main: sonnet + terra + sol + gemini bottom|top)")
print("cleared paperfigures/api_prompts_appendix.tex (merged into main)")
print(f"  sol : bot {sbp:.2f} -> base {sbase:.2f} -> top {stp:.2f}   [{snote}]")
print(f"  gem : bot {gbp:.2f} -> base {gbase:.2f} -> top {gtp:.2f}   [{gnote}]")
