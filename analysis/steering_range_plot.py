"""Steering-range plot: P(y+) under the BOTTOM (min) and TOP (max) extremizer as a
dumbbell, for the 3 API steered cells (100 fresh samples) + open-weight reasoning
dose-response (Qwen3-8B across think budgets, gpt-oss-20B across efforts). A bar
crossing 0.5 = the irrelevant nudge flips the model's modal answer.
"""
import os, json, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

fv = json.load(open("data/final_validate_100.json"))
def api(kb, kt):
    return fv[kb]["p_fresh"], fv[kt]["p_fresh"]

_tsv = list(csv.DictReader(open(os.path.join(OUT, "reasoning_steering_ranges.tsv")), delimiter="\t"))
def owl(model, cell, setting):
    for r in _tsv:
        if r["model"] == model and r["nudge"] == cell and r["setting"] == setting:
            return float(r["bot"]), float(r["top"])
    return None

ROWS = []
for lab, kb, kt in [
        ("Sonnet-5 . low       animals -> conscious", "sonnet-5-low conscious BOT", "sonnet-5-low conscious TOP"),
        ("GPT-5.6-terra . low    typos -> trolley",   "terra-low trolley BOT",       "terra-low trolley TOP"),
        ("Gemini-3-Flash . high  animals -> 5-vs-7",  "gemini-3-flash-high five7 BOT", "gemini-3-flash-high five7 TOP")]:
    b, t = api(kb, kt); ROWS.append((lab, "api", b, t))
for s, lab in [("nothink", "no-think"), ("B256", "256"), ("B512", "512"),
               ("B1024", "1024"), ("B2048", "2048"), ("B4096", "4096")]:
    bt = owl("qwen3_8b", "phrasing_L100_O10_five7", s)
    if bt:
        ROWS.append(("Qwen3-8B . think %8s   phrasing -> 5-vs-7" % lab, "qwen", bt[0], bt[1]))
for s in ["low"]:                        # med/high dropped: parse rate too low to trust
    bt = owl("gptoss_20b", "pool_animals_five7_low", s)
    if bt:
        ROWS.append(("gpt-oss-20B . %6s     animals -> 5-vs-7" % s, "gptoss", bt[0], bt[1]))

nr = len(ROWS)
fig, ax = plt.subplots(figsize=(9.6, 0.40 * nr + 1.4), dpi=150)
for i, (lab, grp, b, t) in enumerate(ROWS):
    y = nr - 1 - i
    lo, hi = min(b, t), max(b, t)
    ax.plot([lo, hi], [y, y], color="#c4c4c4", lw=3.2, solid_capstyle="round", zorder=1)
    ax.scatter([b], [y], s=64, color="#2166ac", zorder=3, edgecolors="white", linewidths=0.5)
    ax.scatter([t], [y], s=64, color="#b2182b", zorder=3, edgecolors="white", linewidths=0.5)
    ax.text(lo - 0.018, y, "%.2f" % lo, ha="right", va="center", fontsize=7.5, color="#555")
    ax.text(hi + 0.018, y, "%.2f" % hi, ha="left", va="center", fontsize=7.5, color="#555")

ax.axvline(0.5, color="k", ls="--", lw=1.0, alpha=0.55, zorder=0)
for i in range(1, nr):
    if ROWS[i][1] != ROWS[i - 1][1]:
        ax.axhline(nr - 1 - i + 0.5, color="#999", lw=0.8, ls=":")

ax.set_yticks(range(nr)); ax.set_yticklabels([r[0] for r in ROWS[::-1]], fontsize=8.5, family="monospace")
ax.set_xlim(-0.10, 1.13); ax.set_ylim(-0.6, nr - 0.6)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("answer probability  $P(y^{+})$", fontsize=10.5)
ax.set_title("Steering range: $P(y^{+})$ under the bottom- vs top-extremizing nudge prompt\n"
             "a bar crossing 0.5 = the irrelevant nudge flips the model's modal answer",
             fontsize=11, pad=8)
ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac", markersize=8, label="bottom extremizer (min $P$)"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor="#b2182b", markersize=8, label="top extremizer (max $P$)")],
          loc="lower right", fontsize=8.5, frameon=True)
ax.grid(axis="x", alpha=0.25)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.savefig(os.path.join(OUT, "steering_ranges.png"), dpi=200, bbox_inches="tight", pad_inches=0.08)
fig.savefig(os.path.join(OUT, "steering_ranges.pdf"), bbox_inches="tight", pad_inches=0.08)
print("wrote %s steering_ranges.{png,pdf} (%d rows)" % (OUT, nr))
