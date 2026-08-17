"""Appendix figure (Section-3 v1 style, item x position, animals200): random-trial cloud
with the optimized extremizer lists overlaid -- distinct-item AND repetition-allowed -- no
tilt band. Shows that under the position-aware model the repetition-allowed extremizers sit
in the same place as the distinct ones: allowing repeats does not reach further.

Random cloud: predicted (a + sum_p B[item_p,p]) vs measured logit for the 20k random 10-item
lists. Extremizers: from {cell}/repeats_ip.json (measured). The item x position ridge is
refit here identically to data/_repeats_ip.py so cloud and extremizers share one axis.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from subliminal.pools import load_pool

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

MODELS = [("qwen25_3b", "Qwen2.5-3B"), ("qwen25_7b", "Qwen2.5-7B"),
          ("qwen25_14b", "Qwen2.5-14B"), ("qwen25_32b", "Qwen2.5-32B"),
          ("llama31_8b", "Llama-3.1-8B"), ("gemma2_9b", "Gemma-2-9B"),
          ("phi4", "Phi-4"), ("olmo2_7b", "OLMo-2-7B")]
CELL = "animals_consider_five7"
CLOUD, DIST, REP = "#9aa7b4", "#2f629e", "#c65c30"     # random cloud, distinct, repeats
INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e6e3"
POOL = load_pool("animals_consider"); NI = len(POOL["items"])


def random_cloud(tag):
    rows = [json.loads(l) for l in open(f"data/cells/{tag}/{CELL}/raw.jsonl")]
    rows = [r for r in rows if abs(r.get("l", 99)) < 30]
    key = "idx" if "idx" in rows[0] else "ch"       # standard cells store the list under "ch"
    LP = len(rows[0][key]); n = len(rows)
    Xip = np.zeros((n, NI * LP)); y = np.array([r["l"] for r in rows], float)
    idx = np.array([r[key] for r in rows])
    for t in range(n):
        Xip[t, idx[t] * LP + np.arange(LP)] = 1
    xm = Xip.mean(0); ym = y.mean()
    b = np.linalg.solve((Xip - xm).T @ (Xip - xm) + 10.0 * np.eye(NI * LP), (Xip - xm).T @ (y - ym))
    a = float(ym - xm @ b); B = b.reshape(NI, LP)
    pred = a + B[idx, np.arange(LP)].sum(1)
    return pred, y


def ex(cond):
    x = list(cond["top"]["pred_by_rank"]) + list(cond["bot"]["pred_by_rank"])
    yy = list(cond["top"]["meas_by_rank"]) + list(cond["bot"]["meas_by_rank"])
    return np.array(x), np.array(yy)


NR, NC = 2, 4
fig, axs2 = plt.subplots(NR, NC, figsize=(3.4 * NC, 3.5 * NR), dpi=200)
axs = axs2.flatten()
rng = np.random.default_rng(0)
for k, (ax, (tag, lab)) in enumerate(zip(axs, MODELS)):
    d = json.load(open(f"data/cells/{tag}/{CELL}/repeats_ip.json"))
    rp, rm = random_cloud(tag)
    sel = rng.choice(len(rp), min(3000, len(rp)), replace=False)
    dx, dy = ex(d["distinct"]); rx, ry = ex(d["repeats"])
    lim = [min(rp.min(), dx.min(), rx.min(), rm.min(), dy.min(), ry.min()) - 0.6,
           max(rp.max(), dx.max(), rx.max(), rm.max(), dy.max(), ry.max()) + 0.6]
    ax.set_facecolor(SURF)
    ax.plot(lim, lim, ls="--", lw=1.0, color=MUTED, alpha=0.5, zorder=1)
    ax.scatter(rp[sel], rm[sel], s=6, color=CLOUD, alpha=0.16, edgecolors="none", zorder=2)
    ax.scatter(dx, dy, s=22, color=DIST, alpha=0.9, edgecolors=SURF, linewidths=0.3, zorder=4)
    ax.scatter(rx, ry, s=22, color=REP, alpha=0.9, edgecolors=SURF, linewidths=0.3, zorder=3, marker="D")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_title(lab, fontsize=10.5, color=INK)
    if k // NC == NR - 1:
        ax.set_xlabel("additive predicted log-odds  $\\hat\\ell$", fontsize=9, color=INK)
    if k % NC == 0:
        ax.set_ylabel("measured log-odds  $\\ell$", fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
fig.patch.set_facecolor(SURF)
axs[NC - 1].legend(handles=[Line2D([0], [0], marker="o", ls="none", mfc=CLOUD, mec="none", ms=6, label="random trials"),
                        Line2D([0], [0], marker="o", ls="none", mfc=DIST, mec="none", ms=7, label="distinct extremizers"),
                        Line2D([0], [0], marker="D", ls="none", mfc=REP, mec="none", ms=6, label="repetition-allowed"),
                        Line2D([0], [0], ls="--", color=MUTED, alpha=0.6, label="$\\ell=\\hat\\ell$")],
               loc="lower right", fontsize=7.5, frameon=False, handletextpad=0.4, labelspacing=0.3)
fig.suptitle("Position-aware model: distinct and repetition-allowed extremizers reach a similar range",
             fontsize=12, color=INK, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.97))

fig.savefig(os.path.join(OUT, "repeats_figure.pdf"), bbox_inches="tight", pad_inches=0.06, facecolor=SURF)
fig.savefig(os.path.join(OUT, "repeats_figure.png"), dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor=SURF)
print("wrote paperfigures/ + figures/ repeats_figure.{pdf,png}  (v1 scatter + repeats extremizers)")
