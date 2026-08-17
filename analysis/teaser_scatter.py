"""Predicted-vs-measured scatter for the animals teaser (fig2_predicted_vs_true).

Qwen2.5-14B, animals_consider / 5-vs-7, item x position fit. Blue cloud = random
prompts; the two dots are the MEASURED-BEST extremizers found across the top-100 /
bottom-100 predicted candidates (s^{*,5} red bottom-left, s^{*,7} green top-right),
plotted at (predicted, measured) log-odds. Reported on the log P(7)/P(5) axis to match
the teaser (so 7-pushing is up/right)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from subliminal.pools import load_pool

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

D = "data/cells/qwen25_14b/animals_consider_five7"
fi = json.load(open(f"{D}/fit_ip.json")); a = float(fi["a"]); B = np.array(fi["beta_ip"])
ex = json.load(open(f"{D}/scatter_extras.json"))
cfg = json.load(open(f"{D}/se_ip_configs.json"))
ITEMS = load_pool("animals_consider")["items"]

# the two displayed extremizers (short, still >99%); exact ordered lists (item x position)
FIVE = ["ladybug", "blue whale", "bobcat", "elephant", "manta ray", "cod", "minnow", "tuna", "condor", "ant"]
SEVEN = ["sloth", "magpie", "orca", "hornet", "wasp", "zebra", "giraffe", "locust", "cricket", "tasmanian devil"]

# random cloud: item x position predicted vs measured (codebase l = log P(5)/P(7))
pred, meas = [], []
for line in open(f"{D}/raw.jsonl"):
    r = json.loads(line)
    if "l" not in r or abs(r["l"]) >= 30:
        continue
    ch = r.get("ch", r.get("idx"))
    pred.append(a + float(sum(B[p, c] for p, c in enumerate(ch))))
    meas.append(float(r["l"]))
pred = np.array(pred); meas = np.array(meas)

# look up the two displayed configs by their exact ordered animal lists
def find_cfg(side, target):
    chs, meas, pred = cfg[side]["chs"], ex[side]["meas"], ex[side]["pred"]
    for i, ch in enumerate(chs):
        if [ITEMS[j] for j in ch] == target:
            return pred[i], meas[i]
    raise ValueError("config not found: " + ", ".join(target))
five = find_cfg("top", FIVE)     # 5-pusher (codebase pred, meas = log P(5)/P(7))
seven = find_cfg("bot", SEVEN)   # 7-pusher

# flip to log P(7)/P(5): negate both axes
flip = lambda xy: (-xy[0], -xy[1])
px, py = -pred, -meas
fx, fy = flip(five)        # bottom-left
sx, sy = flip(seven)       # top-right

fig, ax = plt.subplots(figsize=(5.2, 2.05), dpi=200)
ax.scatter(px, py, s=3, alpha=0.06, color="#1f77b4", edgecolors="none",
           rasterized=True, label="random prompts")
lo = min(px.min(), fx, sx) - 0.3; hi = max(px.max(), fx, sx) + 0.3
ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1.1, label="y=x", zorder=1)
ax.scatter([sx], [sy], s=15, color="#2a8a3a", edgecolors="none", zorder=5)  # s*,7
ax.scatter([fx], [fy], s=15, color="#b3251d", edgecolors="none", zorder=5)  # s*,5
ax.set_xlabel("additive predicted log-odds", fontsize=11)
ax.set_ylabel("log-odds", fontsize=11)
ax.tick_params(labelsize=9)
leg = ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9,
                borderpad=0.3, handletextpad=0.4)
for hnd in leg.legend_handles:          # cloud dots are near-transparent; the legend key must be opaque
    hnd.set_alpha(1.0)
    if hasattr(hnd, "set_sizes"):
        hnd.set_sizes([22])
fig.tight_layout(pad=0.3)
for e in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig2_predicted_vs_true.{e}"), bbox_inches="tight", pad_inches=0.02)
print(f"wrote figures/fig2_predicted_vs_true.pdf/.png")
print(f"  s*,5 (red, bottom-left): x={fx:.2f} y={fy:.2f}   s*,7 (green, top-right): x={sx:.2f} y={sy:.2f}")
