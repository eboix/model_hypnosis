"""Two summary figures for the transfer section (sec 4).

  transfer_summary.png    : cross-model FOUND-extremizer transfer (4 panels)
      A. mode-flip rate by search budget (δ1 vs δ2), by flippability subset
      B. directional flip on both-flippable pairs, per nudge
         (source top-100 -> target y+ ; source bottom-100 -> target y-)
      C. found-pair DIRECTION: fraction of pairs with ℓ_T(s_top) > ℓ_T(s_bot)
         (order preserved) vs < (reversed), per nudge  [diverging]
      D. average fraction of the target's own range steered (δ1, δ2): flippable vs overall

  reasoning_switch_transfer.png : steering across the Qwen3-8B reasoning switch
      spread P(top)-P(bot) vs thinking budget, for nothink-fit and think-fit prompts
      evaluated in each mode; native baselines stay high, cross-transfer decays.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import transfer_found_analysis as T

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID, SURF = "#1a1c1f", "#5c626b", "#e4e4e1", "#ffffff"
BLUE, ORANGE = "#2f629e", "#c65c30"      # validated muted categorical pair
RED, COOL = "#b3251d", "#1f4e9c"
GREENED, GREY = "#2a9d3f", "#8b9099"
NUD = [("animals_consider", "animals"), ("phrasing_L20_O10", "phrasing"),
       ("jsonblob", "JSON"), ("typos", "typos")]


def get_recs():
    _, _, recs = T.analyze()
    recs = [r for r in recs if not any(np.isnan(x) for x in r[3:13])]
    return recs


def cross_model_fig(recs):
    nu = np.array([r[0] for r in recs])
    sf = np.array([r[3] for r in recs], bool); tf = np.array([r[4] for r in recs], bool)
    f1 = np.array([r[5] for r in recs]); f2 = np.array([r[6] for r in recs])
    d1 = np.array([r[7] for r in recs]); d2 = np.array([r[8] for r in recs])
    topf = np.array([r[9] for r in recs]); botf = np.array([r[10] for r in recs])
    d1pos = d1 > 0; d1neg = d1 < 0
    both = sf & tf

    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.6), dpi=150)
    for ax in axs.flat:
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9.5)

    # --- A: flip rate by subset (delta1 vs delta2) ---
    ax = axs[0, 0]
    subs = [("all\noff-diagonal", np.ones(len(recs), bool)),
            ("target\nflippable", tf), ("both\nflippable", both)]
    x = np.arange(len(subs)); w = 0.38
    v1 = [100 * f1[m].mean() for _, m in subs]; v2 = [100 * f2[m].mean() for _, m in subs]
    ax.bar(x - w / 2, v1, w, color=ORANGE, label="δ₁  (found pair, 2 queries)")
    ax.bar(x + w / 2, v2, w, color=BLUE, label="δ₂  (best of top/bottom-100)")
    for xi, (a, b) in enumerate(zip(v1, v2)):
        ax.text(xi - w / 2, a + 1, f"{a:.0f}", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2, b + 1, f"{b:.0f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in subs])
    ax.set_ylabel("mode-flip rate  (%)", fontsize=10, color=INK); ax.set_ylim(0, 65)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")
    ax.set_title("A · How often steering flips the target's answer", fontsize=11, color=INK, loc="left")

    # --- B: directional flip on both-flippable, per nudge ---
    ax = axs[0, 1]
    labs = [l for _, l in NUD]; x = np.arange(len(NUD)); w = 0.27
    tp = [100 * topf[both & (nu == n)].mean() for n, _ in NUD]
    bt = [100 * botf[both & (nu == n)].mean() for n, _ in NUD]
    bo = [100 * ((topf > 0) & (botf > 0))[both & (nu == n)].mean() for n, _ in NUD]
    ax.bar(x - w, tp, w, color=RED, label="top-100 → y⁺")
    ax.bar(x, bt, w, color=COOL, label="bottom-100 → y⁻")
    ax.bar(x + w, bo, w, color="#3a3f48", label="both directions")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("success rate on both-flippable  (%)", fontsize=10, color=INK); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.8, loc="upper right", ncol=1)
    ax.set_title("B · Does the source's extreme drive the target each way?", fontsize=11, color=INK, loc="left")

    # --- C: found-pair direction preserved vs reversed (diverging) ---
    ax = axs[1, 0]
    rows = [("both-flippable\n(all)", both)] + [(l, both & (nu == n)) for n, l in NUD]
    y = np.arange(len(rows))[::-1]
    pos = [100 * d1pos[m].mean() for _, m in rows]; neg = [100 * d1neg[m].mean() for _, m in rows]
    ax.barh(y, pos, color=BLUE, height=0.6, label="direction preserved  (ℓ$_T$: top > bottom)")
    ax.barh(y, [-v for v in neg], color=RED, height=0.6, label="direction reversed")
    for yi, (p, n) in zip(y, zip(pos, neg)):
        ax.text(p + 2, yi, f"{p:.0f}%", va="center", fontsize=8.6, color=BLUE)
        ax.text(-n - 2, yi, f"{n:.0f}%", va="center", ha="right", fontsize=8.6, color=RED)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([l for l, _ in rows], fontsize=9)
    ax.set_xlim(-58, 112); ax.set_xticks([-40, 0, 40, 80]); ax.set_xticklabels(["40", "0", "40", "80"])
    ax.set_xlabel("% of transfer pairs", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.4, loc="upper right")
    ax.set_title("C · Found-pair direction transfers even when it doesn't flip", fontsize=11, color=INK, loc="left")

    # --- D: fraction of target range steered ---
    ax = axs[1, 1]
    subs = [("all\noff-diagonal", np.ones(len(recs), bool)),
            ("target\nflippable", tf), ("both\nflippable", both)]
    x = np.arange(len(subs)); w = 0.38
    m1 = [d1[m].mean() for _, m in subs]; m2 = [d2[m].mean() for _, m in subs]
    ax.bar(x - w / 2, m1, w, color=ORANGE, label="δ₁ / range$_T$")
    ax.bar(x + w / 2, m2, w, color=BLUE, label="δ₂ / range$_T$")
    for xi, (a, b) in enumerate(zip(m1, m2)):
        ax.text(xi - w / 2, a + .01, f"{a:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2, b + .01, f"{b:.2f}", ha="center", fontsize=9, color=INK)
    ax.axhline(1.0, color=GREY, lw=1, ls=":")
    ax.text(2.42, 1.02, "target's full range", fontsize=8, color=GREY, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in subs])
    ax.set_ylabel("mean fraction of target range steered", fontsize=10, color=INK); ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")
    ax.set_title("D · How much of the target's range the found nudge reaches", fontsize=11, color=INK, loc="left")

    fig.suptitle("Cross-model transfer of found nudge extremizers  (16 source × 16 target models, 4 nudges × 3 effects)",
                 fontsize=13, color=INK, y=0.99)
    fig.patch.set_facecolor(SURF)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(os.path.join(OUT, "transfer_summary.png"), dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(os.path.join(OUT, "transfer_summary.pdf"), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def target_random_logits(tag, nu, ef):
    p = f"data/cells/{tag}/{nu}_{ef}/raw.jsonl"
    if not os.path.exists(p):
        return None
    L = [r["l"] for r in (json.loads(l) for l in open(p)) if "l" in r and abs(r["l"]) < 30]
    return np.array(L) if len(L) >= 200 else None


def random_baseline(nu, ef, res, B=500, K=100):
    """δ̃₂ null: max of K random prompts − min of K random prompts on each target,
    normalized by that target's obtainable range. Returns fracs over targets."""
    g = np.random.default_rng(0); fr = []
    for j, tag in enumerate(T.TAGS):
        L = target_random_logits(tag, nu, ef); rng = res["rng"][j]
        if L is None or np.isnan(rng) or rng == 0:
            continue
        mx = L[g.integers(0, len(L), (B, K))].max(1)
        mn = L[g.integers(0, len(L), (B, K))].min(1)
        fr.append(float(np.median(mx - mn)) / rng)
    return np.array(fr)


def transfer_dist_fig(results):
    """Per nudge x effect: distribution of the off-diagonal logit transfer δ₂/range_T
    across all 16x15 source->target pairs, with the median marked (δ₁ median overlaid),
    plus the random-100/random-100 baseline δ̃₂ over targets."""
    NC = {"animals_consider": "#2a9d3f", "phrasing_L20_O10": "#255ea6",
          "jsonblob": "#e8790c", "typos": "#8e44ad"}
    rows = []                          # (label, nudge, d2vals, d1vals, rand_fracs)
    print(f"\n{'cell':22s}  transfer δ₂  δ₁-pair   random δ̃₂  (medians, /range_T)")
    for nu, ndl in NUD:
        for ef, efl in T.EFFS:
            res = results.get((nu, ef))
            if res is None:
                continue
            M2 = res["M"]["d2"]; M1 = res["M"]["d1"]
            n = M2.shape[0]; off = ~np.eye(n, dtype=bool)
            d2 = M2[off]; d2 = d2[~np.isnan(d2)]
            d1 = M1[off]; d1 = d1[~np.isnan(d1)]
            rb = random_baseline(nu, ef, res)
            rows.append((f"{ndl} → {efl}", nu, d2, d1, rb))
            print(f"  {ndl+' → '+efl:20s}    {np.median(d2):.2f}       {np.median(d1):.2f}"
                  f"       {np.median(rb):.2f}")

    fig, ax = plt.subplots(figsize=(9.2, 8.4), dpi=150)
    ax.set_facecolor(SURF)
    y = []; ypos = 0.0; ticks = []; ticklab = []
    prev_nu = None
    for lab, nu, d2, d1, rb in rows:
        if prev_nu is not None and nu != prev_nu:
            ypos -= 0.7                 # gap between nudge groups
        ypos -= 1.0; prev_nu = nu
        col = NC[nu]
        vp = ax.violinplot([d2], positions=[ypos], vert=False, widths=0.82,
                           showmeans=False, showextrema=False, showmedians=False)
        for b in vp["bodies"]:
            b.set_facecolor(col); b.set_edgecolor("none"); b.set_alpha(0.32)
        med2 = float(np.median(d2)); med1 = float(np.median(d1))
        ax.plot([med2, med2], [ypos - 0.34, ypos + 0.34], color=col, lw=2.4, zorder=5)
        ax.scatter([med1], [ypos + 0.24], marker="D", s=24, color="#3a3f48", zorder=6)
        # random-100 baseline: median over targets + [P25,P75] whisker
        rmed = float(np.median(rb)); r25, r75 = np.percentile(rb, [25, 75])
        ax.plot([r25, r75], [ypos - 0.26, ypos - 0.26], color=GREY, lw=1.4, zorder=5)
        ax.scatter([rmed], [ypos - 0.26], marker="|", s=90, color="#4a4f57", lw=1.8, zorder=6)
        ax.text(max(np.percentile(d2, 97), 1.05) + 0.03, ypos, f"med {med2:.2f}",
                va="center", fontsize=8.3, color=col)
        ticks.append(ypos); ticklab.append(lab)
    ax.axvline(1.0, color=GREY, lw=1.1, ls=":")
    ax.text(1.0, ypos - 1.1, "target's\nfull range", fontsize=8, color=GREY, ha="center", va="top")
    ax.axvline(0.0, color=MUTED, lw=1)
    ax.set_yticks(ticks); ax.set_yticklabels(ticklab, fontsize=9.5)
    ax.set_xlim(-0.05, 1.5); ax.set_xlabel("logit transfer   δ₂ / range$_T$   (per source→target pair)",
                                           fontsize=10.5, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.set_title("Distribution of cross-model logit transfer, per nudge × effect\n"
                 "violin = transferred δ₂/range$_T$ over 240 off-diagonal pairs",
                 fontsize=12, color=INK, loc="left")
    handles = [Line2D([0], [0], color="#555", lw=2.4, label="transfer δ₂ median"),
               Line2D([0], [0], marker="D", ls="none", mfc="#3a3f48", mec="none", ms=6,
                      label="δ₁ found-pair median"),
               Line2D([0], [0], marker="|", ls="none", mec="#4a4f57", mew=1.8, ms=11,
                      label="random-100 δ̃₂ baseline (median, P25–75 over targets)")]
    ax.legend(handles=handles, frameon=False, fontsize=8.8, loc="upper center",
              bbox_to_anchor=(0.5, -0.075), ncol=3, handletextpad=0.4, columnspacing=1.4)
    fig.patch.set_facecolor(SURF); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transfer_logit_dist.png"), dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(os.path.join(OUT, "transfer_logit_dist.pdf"), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def delta1_sign_fig(results):
    """Per nudge x effect: fraction of off-diagonal pairs where the found pair's direction
    is PRESERVED on the target (δ₁ = ℓ_T(s_top)−ℓ_T(s_bot) > 0) vs REVERSED (< 0)."""
    NC = {"animals_consider": "#2a9d3f", "phrasing_L20_O10": "#255ea6",
          "jsonblob": "#e8790c", "typos": "#8e44ad"}
    print(f"\n{'cell':22s}  δ₁>0 (preserved)  δ₁<0 (reversed)   n")
    rows = []
    for nu, ndl in NUD:
        for ef, efl in T.EFFS:
            res = results.get((nu, ef))
            if res is None:
                continue
            M1 = res["M"]["d1"]; n = M1.shape[0]; off = ~np.eye(n, dtype=bool)
            v = M1[off]; v = v[~np.isnan(v)]
            pos = 100 * np.mean(v > 0); neg = 100 * np.mean(v < 0)
            rows.append((f"{ndl} → {efl}", nu, pos, neg))
            print(f"  {ndl+' → '+efl:20s}     {pos:5.1f}%           {neg:5.1f}%       {len(v)}")

    fig, ax = plt.subplots(figsize=(8.8, 8.0), dpi=150)
    ax.set_facecolor(SURF)
    ypos = 0.0; ticks = []; ticklab = []; prev = None
    for lab, nu, pos, neg in rows:
        if prev is not None and nu != prev:
            ypos -= 0.7
        ypos -= 1.0; prev = nu
        ax.barh(ypos, pos, color=BLUE, height=0.72, zorder=3)
        ax.barh(ypos, -neg, color=RED, height=0.72, zorder=3)
        ax.text(pos + 1.5, ypos, f"{pos:.0f}%", va="center", fontsize=8.8, color=BLUE)
        ax.text(-neg - 1.5, ypos, f"{neg:.0f}%", va="center", ha="right", fontsize=8.8, color=RED)
        ax.scatter([-62], [ypos], color=NC[nu], s=34, zorder=4)   # nudge colour tag
        ticks.append(ypos); ticklab.append(lab)
    ax.axvline(0, color=MUTED, lw=1.2)
    ax.set_yticks(ticks); ax.set_yticklabels(ticklab, fontsize=9.5)
    ax.set_xlim(-68, 108); ax.set_xticks([-40, 0, 40, 80]); ax.set_xticklabels(["40", "0", "40", "80"])
    ax.set_xlabel("% of off-diagonal source→target pairs", fontsize=10.5, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.set_title("Does the found pair keep its direction on the target?  (sign of δ₁)\n"
                 "δ₁ = ℓ$_T$(source-top) − ℓ$_T$(source-bottom)   ·   blue = preserved (>0), red = reversed (<0)",
                 fontsize=12, color=INK, loc="left")
    ax.text(80, ticks[0] + 1.0, "preserved →", fontsize=9, color=BLUE, ha="right", fontweight="bold")
    ax.text(-40, ticks[0] + 1.0, "← reversed", fontsize=9, color=RED, ha="left", fontweight="bold")
    fig.patch.set_facecolor(SURF); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transfer_delta1_sign.png"), dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(os.path.join(OUT, "transfer_delta1_sign.pdf"), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def delta1_subset_fig(results):
    """Two-panel dumbbell on δ₁, per nudge x effect, comparing ALL off-diagonal vs
    BOTH-flippable pairs:  (left) direction-preserved %,  (right) median δ₁/range_T."""
    NC = {"animals_consider": "#2a9d3f", "phrasing_L20_O10": "#255ea6",
          "jsonblob": "#e8790c", "typos": "#8e44ad"}
    C_ALL, C_BF = "#2f629e", "#c65c30"
    rows = []
    for nu, ndl in NUD:
        for ef, efl in T.EFFS:
            res = results.get((nu, ef))
            if res is None:
                continue
            M1 = res["M"]["d1"]; n = M1.shape[0]; off = ~np.eye(n, dtype=bool)
            sf = res["src_flip"]; tf = res["tgt_flip"]
            both = (sf[:, None] == 1) & (tf[None, :] == 1) & off
            vA = M1[off]; vA = vA[~np.isnan(vA)]
            vB = M1[both]; vB = vB[~np.isnan(vB)]
            rows.append((f"{ndl} → {efl}", nu, vA, vB, len(vB)))

    BK = "#111"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.9), dpi=150)
    OFF = 0.2; BH = 0.36
    ekw = dict(ecolor=BK, elinewidth=1.1, capsize=2.5)
    ypos = 0.0; ticks = []; ticklab = []; prev = None
    for lab, nu, vA, vB, nb in rows:
        if prev is not None and nu != prev:
            ypos -= 0.5
        ypos -= 1.0; prev = nu
        haveB = len(vB) > 0
        # LEFT: sign preserved %  (binomial SE)
        pa = np.mean(vA > 0); pb = np.mean(vB > 0) if haveB else np.nan
        sea = 100 * np.sqrt(pa * (1 - pa) / len(vA))
        seb = 100 * np.sqrt(pb * (1 - pb) / len(vB)) if haveB else 0
        axL.barh(ypos + OFF, 100 * pa, BH, color=C_ALL, xerr=sea, error_kw=ekw, zorder=3)
        axL.text(1.5, ypos + OFF, "n=240", va="center", ha="left", fontsize=8.5, color="white", zorder=5)
        if haveB:
            axL.barh(ypos - OFF, 100 * pb, BH, color=C_BF, xerr=seb, error_kw=ekw, zorder=3)
            axL.text(1.5, ypos - OFF, f"n={nb}", va="center", ha="left", fontsize=8.5, color="white", zorder=5)
        # RIGHT: mean range fraction  (SE)
        fa = np.mean(vA); fb = np.mean(vB) if haveB else np.nan
        sfa = np.std(vA) / np.sqrt(len(vA)); sfb = np.std(vB) / np.sqrt(len(vB)) if haveB else 0
        axR.barh(ypos + OFF, fa, BH, color=C_ALL, xerr=sfa, error_kw=ekw, zorder=3)
        if haveB:
            axR.barh(ypos - OFF, fb, BH, color=C_BF, xerr=sfb, error_kw=ekw, zorder=3)
        ticks.append(ypos); ticklab.append(lab)
    axL.axvline(50, color=BK, lw=1.0, ls="--")
    axL.text(50, ticks[-1] - 0.75, "chance", fontsize=10.5, color=BK, ha="center")
    axR.axvline(0, color=BK, lw=1.0)
    axL.set_yticks(ticks); axL.set_yticklabels(ticklab, fontsize=13, color=BK)
    axR.set_yticks(ticks); axR.set_yticklabels([])
    axL.set_xlim(0, 104); axL.set_xticks([0, 25, 50, 75, 100])
    axR.set_xlim(-0.03, 0.66); axR.set_xticks([0, 0.2, 0.4, 0.6])
    axL.set_xlabel("direction preserved on target  (% of prompt pairs)", fontsize=12.5, color=BK)
    axR.set_xlabel("mean fraction of target range steered", fontsize=12.5, color=BK)
    for ax in (axL, axR):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BK)
        ax.tick_params(colors=BK, labelsize=11.5)
    axL.set_title("Does the source pair of prompts keep its direction?", fontsize=13.5, color=BK, loc="center")
    axR.set_title("How far the source pair of prompts steers the target", fontsize=13.5, color=BK, loc="center")
    handles = [Patch(facecolor=C_ALL, label="all source→target pairs of models"),
               Patch(facecolor=C_BF, label="only source→target pairs of models where both are flippable by this nudge × effect")]
    fig.legend(handles=handles, frameon=False, fontsize=11, ncol=1, loc="upper center",
               bbox_to_anchor=(0.5, 0.955), handletextpad=0.5, labelspacing=0.4)
    fig.suptitle("Transfer of found extremizing prompts to other models",
                 fontsize=15, color=BK, y=0.995)
    fig.patch.set_facecolor(SURF); fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(OUT, "transfer_delta1_subsets.png"), dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(os.path.join(OUT, "transfer_delta1_subsets.pdf"), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def model_pair_fig(results):
    """Model x model transfer heatmap: mean found-pair transfer (fraction of target range)
    averaged over all 12 nudge x effect cells. rows = source, cols = target, family-ordered."""
    BK = "#111"
    LBL = {"qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
           "qwen25_32b": "Qwen2.5-32B", "qwen25_72b": "Qwen2.5-72B", "qwen3_4b": "Qwen3-4B",
           "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B",
           "qwen35_9b": "Qwen3.5-9B", "gemma2_9b": "Gemma-2-9B", "gemma4_12b": "Gemma-4-12B",
           "llama31_8b": "Llama-3.1-8B", "phi4": "Phi-4", "olmo2_7b": "OLMo-2-7B", "olmo3_7b": "OLMo-3-7B"}
    order = T.TAGS; labs = [LBL.get(t, t) for t in order]
    mats = [results[(nu, ef)]["M"]["d1"] for nu, _ in T.NUD for ef, _ in T.EFFS if (nu, ef) in results]
    M = np.nanmean(np.stack(mats), 0); np.fill_diagonal(M, np.nan)

    fig, ax = plt.subplots(figsize=(9.4, 8.4), dpi=150)
    cmap = mpl.colormaps["Blues"].copy(); cmap.set_bad("#eeede9")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=0.30, aspect="equal")
    for i in range(len(order)):
        for j in range(len(order)):
            if not np.isnan(M[i, j]) and M[i, j] >= 0.22:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=10.5, color=BK)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labs, fontsize=10.5, color=BK)
    ax.set_xlabel("target model", fontsize=13, color=BK)
    ax.set_ylabel("source model  (nudge found on)", fontsize=13, color=BK)
    for b in (4.5, 8.5, 9.5, 11.5, 12.5, 13.5):
        ax.axhline(b, color=BK, lw=0.8); ax.axvline(b, color=BK, lw=0.8)
    ax.tick_params(colors=BK, length=0)
    for s in ax.spines.values():
        s.set_color(BK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, extend="max")
    cb.set_label("mean found-pair transfer  (fraction of target range, avg over 12 cells)",
                 fontsize=11.5, color=BK)
    cb.ax.tick_params(colors=BK, labelsize=10.5)
    ax.set_title("Which models transfer to which?  (found extremizing pair, source → target)",
                 fontsize=14.5, color=BK, pad=10)
    fig.patch.set_facecolor(SURF); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transfer_model_pairs.png"), dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(os.path.join(OUT, "transfer_model_pairs.pdf"), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    # print rankings
    Ms = (M + M.T) / 2
    sp = sorted([(i, j) for i in range(len(order)) for j in range(i + 1, len(order))],
                key=lambda p: -Ms[p])
    print("\nTOP MUTUAL model pairs (avg both directions, found-pair fraction of range):")
    for i, j in sp[:10]:
        print(f"  {LBL[order[i]]:13s} <-> {LBL[order[j]]:13s}  {Ms[i, j]:.3f}")


def model_pair_sign_fig(results):
    """Model x model heatmap shaded by the FRACTION of the 12 nudge x effect settings in which
    the found pair keeps its direction on the target (δ₁ > 0). Diverging, centered at chance=0.5."""
    BK = "#111"
    LBL = {"qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
           "qwen25_32b": "Qwen2.5-32B", "qwen25_72b": "Qwen2.5-72B", "qwen3_4b": "Qwen3-4B",
           "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B",
           "qwen35_9b": "Qwen3.5-9B", "gemma2_9b": "Gemma-2-9B", "gemma4_12b": "Gemma-4-12B",
           "llama31_8b": "Llama-3.1-8B", "phi4": "Phi-4", "olmo2_7b": "OLMo-2-7B", "olmo3_7b": "OLMo-3-7B"}
    order = T.TAGS; labs = [LBL.get(t, t) for t in order]; n = len(order)
    mats = [results[(nu, ef)]["M"]["d1"] for nu, _ in T.NUD for ef, _ in T.EFFS if (nu, ef) in results]
    stack = np.stack(mats)                                   # (12, 16, 16)
    S = np.mean(stack > 0, axis=0).astype(float)             # fraction of settings with δ₁>0
    # diagonal stays = 1.0 (self)
    fig, ax = plt.subplots(figsize=(9.4, 8.4), dpi=150)
    cmap = mpl.colormaps["RdBu"].copy(); cmap.set_bad("#eeede9")
    norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    im = ax.imshow(S, cmap=cmap, norm=norm, aspect="equal")
    for i in range(n):
        for j in range(n):
            if i != j and (S[i, j] >= 0.83 or S[i, j] <= 0.34):
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=10.5, color=BK)
    ax.set_yticks(range(n)); ax.set_yticklabels(labs, fontsize=10.5, color=BK)
    ax.set_xlabel("target model", fontsize=13, color=BK)
    ax.set_ylabel("source model  (nudge found on)", fontsize=13, color=BK)
    for b in (4.5, 8.5, 9.5, 11.5, 12.5, 13.5):
        ax.axhline(b, color=BK, lw=0.8); ax.axvline(b, color=BK, lw=0.8)
    ax.tick_params(colors=BK, length=0)
    for s in ax.spines.values():
        s.set_color(BK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("fraction of the 12 settings where the found pair keeps its direction",
                 fontsize=11.5, color=BK)
    cb.ax.tick_params(colors=BK, labelsize=10.5)
    cb.ax.axhline(0.5, color=BK, lw=1.0)   # mark chance on the bar
    ax.set_title("How consistently does the found pair keep its direction?  (source → target)",
                 fontsize=14.5, color=BK, pad=10)
    fig.patch.set_facecolor(SURF); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transfer_model_pairs_sign.png"), dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(os.path.join(OUT, "transfer_model_pairs_sign.pdf"), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    Ms = (S + S.T) / 2
    sp = sorted([(i, j) for i in range(n) for j in range(i + 1, n)], key=lambda p: -Ms[p])
    print("\nTOP MUTUAL pairs by sign-consistency (avg both dirs, frac of 12 settings δ₁>0):")
    for i, j in sp[:8]:
        print(f"  {LBL[order[i]]:13s} <-> {LBL[order[j]]:13s}  {Ms[i, j]:.2f}")


def sign_combined_fig(results):
    """Side-by-side sign figure for the text: (left) model x model fraction-of-settings the
    source pair keeps its direction; (right) direction-preserved % per nudge x effect,
    all off-diagonal vs both-flippable."""
    BK = "#111"; C_ALL, C_BF = BLUE, ORANGE
    LBL = {"qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
           "qwen25_32b": "Qwen2.5-32B", "qwen25_72b": "Qwen2.5-72B", "qwen3_4b": "Qwen3-4B",
           "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B",
           "qwen35_9b": "Qwen3.5-9B", "gemma2_9b": "Gemma-2-9B", "gemma4_12b": "Gemma-4-12B",
           "llama31_8b": "Llama-3.1-8B", "phi4": "Phi-4", "olmo2_7b": "OLMo-2-7B", "olmo3_7b": "OLMo-3-7B"}
    order = T.TAGS; labs = [LBL.get(t, t) for t in order]; n = len(order)
    mats = [results[(nu, ef)]["M"]["d1"] for nu, _ in T.NUD for ef, _ in T.EFFS if (nu, ef) in results]
    S = np.mean(np.stack(mats) > 0, axis=0).astype(float)     # diagonal = 1.0

    fig = plt.figure(figsize=(15.2, 6.7), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.30)
    axH = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])

    # ---- LEFT: model x model heatmap ----
    cmap = mpl.colormaps["RdBu"].copy()
    norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    im = axH.imshow(S, cmap=cmap, norm=norm, aspect="auto")
    for i in range(n):
        for j in range(n):
            if i != j and (S[i, j] >= 0.83 or S[i, j] <= 0.34):
                axH.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=6.4, color="white")
    axH.set_xticks(range(n)); axH.set_xticklabels(labs, rotation=45, ha="right", fontsize=8.5, color=BK)
    axH.set_yticks(range(n)); axH.set_yticklabels(labs, fontsize=8.5, color=BK)
    axH.set_xlabel("target model", fontsize=11, color=BK)
    axH.set_ylabel("source model  (prompts found on)", fontsize=11, color=BK)
    for b in (4.5, 8.5, 9.5, 11.5, 12.5, 13.5):
        axH.axhline(b, color=BK, lw=0.7); axH.axvline(b, color=BK, lw=0.7)
    axH.tick_params(colors=BK, length=0)
    for s in axH.spines.values():
        s.set_color(BK)
    axH.set_title("By source → target model pair", fontsize=13, color=BK, pad=8)
    cb = fig.colorbar(im, ax=axH, fraction=0.045, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("fraction of settings preserved", fontsize=10, color=BK)
    cb.ax.tick_params(colors=BK, labelsize=9); cb.ax.axhline(0.5, color=BK, lw=1.0)

    # ---- RIGHT: direction-preserved % per nudge x effect, two subsets ----
    OFF = 0.2; BH = 0.36; ekw = dict(ecolor=BK, elinewidth=1.0, capsize=2.3)
    ypos = 0.0; ticks = []; ticklab = []; prev = None
    for nu, ndl in NUD:
        for ef, efl in T.EFFS:
            res = results.get((nu, ef)); M1 = res["M"]["d1"]; off = ~np.eye(n, dtype=bool)
            sf = res["src_flip"]; tf = res["tgt_flip"]
            both = (sf[:, None] == 1) & (tf[None, :] == 1) & off
            vA = M1[off]; vA = vA[~np.isnan(vA)]; vB = M1[both]; vB = vB[~np.isnan(vB)]
            pa = np.mean(vA > 0); pb = np.mean(vB > 0) if len(vB) else np.nan
            sea = 100 * np.sqrt(pa * (1 - pa) / len(vA))
            seb = 100 * np.sqrt(pb * (1 - pb) / len(vB)) if len(vB) else 0
            if prev is not None and nu != prev:
                ypos -= 0.5
            ypos -= 1.0; prev = nu
            axB.barh(ypos + OFF, 100 * pa, BH, color=C_ALL, xerr=sea, error_kw=ekw, zorder=3)
            axB.text(1.5, ypos + OFF, "n=240", va="center", ha="left", fontsize=6.6, color="white", zorder=5)
            if len(vB):
                axB.barh(ypos - OFF, 100 * pb, BH, color=C_BF, xerr=seb, error_kw=ekw, zorder=3)
                axB.text(1.5, ypos - OFF, f"n={len(vB)}", va="center", ha="left", fontsize=6.6, color="white", zorder=5)
            ticks.append(ypos); ticklab.append(f"{ndl} → {efl}")
    axB.axvline(50, color=BK, lw=1.0, ls="--")
    axB.text(50, ticks[-1] - 0.8, "chance", fontsize=9, color=BK, ha="center")
    axB.set_yticks(ticks); axB.set_yticklabels(ticklab, fontsize=10.5, color=BK)
    axB.yaxis.tick_right(); axB.tick_params(axis="y", length=0)   # labels on outer edge, no tick dashes
    axB.set_xlim(0, 104); axB.set_xticks([0, 25, 50, 75, 100])
    axB.set_xlabel("direction preserved on target  (% of prompt pairs)", fontsize=11, color=BK)
    axB.set_title("By cue × effect", fontsize=13, color=BK, pad=8)
    axB.legend(handles=[Patch(facecolor=C_ALL, label="all source→target model pairs"),
                        Patch(facecolor=C_BF, label="only pairs where both are flippable")],
               frameon=False, fontsize=9.5, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axB.spines[s].set_color(BK)
    axB.tick_params(colors=BK, labelsize=10.5)

    fig.suptitle("How consistently does the source pair of prompts keep its direction on the target?",
                 fontsize=14.5, color=BK, y=1.0)
    fig.patch.set_facecolor(SURF); fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(OUT, "transfer_sign_combined.png"), dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(os.path.join(OUT, "transfer_sign_combined.pdf"), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def range_fraction_fig(results):
    """Appendix: mean fraction of the target's obtainable range steered by the source pair,
    per nudge x effect, all off-diagonal vs both-flippable."""
    BK = "#111"; C_ALL, C_BF = BLUE, ORANGE
    n = len(T.TAGS)
    fig, ax = plt.subplots(figsize=(8.2, 7.4), dpi=150)
    ax.set_facecolor(SURF)
    OFF = 0.2; BH = 0.36; ekw = dict(ecolor=BK, elinewidth=1.0, capsize=2.5)
    ypos = 0.0; ticks = []; ticklab = []; prev = None
    for nu, ndl in NUD:
        for ef, efl in T.EFFS:
            res = results.get((nu, ef)); M1 = res["M"]["d1"]; off = ~np.eye(n, dtype=bool)
            sf = res["src_flip"]; tf = res["tgt_flip"]
            both = (sf[:, None] == 1) & (tf[None, :] == 1) & off
            vA = M1[off]; vA = vA[~np.isnan(vA)]; vB = M1[both]; vB = vB[~np.isnan(vB)]
            fa = np.mean(vA); fb = np.mean(vB) if len(vB) else np.nan
            sfa = np.std(vA) / np.sqrt(len(vA)); sfb = np.std(vB) / np.sqrt(len(vB)) if len(vB) else 0
            if prev is not None and nu != prev:
                ypos -= 0.5
            ypos -= 1.0; prev = nu
            ax.barh(ypos + OFF, fa, BH, color=C_ALL, xerr=sfa, error_kw=ekw, zorder=3)
            if len(vB):
                ax.barh(ypos - OFF, fb, BH, color=C_BF, xerr=sfb, error_kw=ekw, zorder=3)
                ax.text(fb + sfb + 0.008, ypos - OFF, f"n={len(vB)}", va="center", ha="left",
                        fontsize=7.5, color=BK)
            ticks.append(ypos); ticklab.append(f"{ndl} → {efl}")
    ax.axvline(0, color=BK, lw=1.0)
    ax.set_yticks(ticks); ax.set_yticklabels(ticklab, fontsize=11, color=BK)
    ax.set_xlim(-0.03, 0.66); ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("mean fraction of target range steered", fontsize=12, color=BK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BK)
    ax.tick_params(colors=BK, labelsize=11)
    ax.legend(handles=[Patch(facecolor=C_ALL, label="all source→target model pairs (n=240)"),
                       Patch(facecolor=C_BF, label="only pairs where both are flippable")],
              frameon=False, fontsize=10, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.075))
    fig.suptitle("How much of the target's range the source pair of prompts steers",
                 fontsize=13.5, color=BK, y=0.99)
    fig.patch.set_facecolor(SURF); fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(OUT, "transfer_range_fraction.png"), dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(os.path.join(OUT, "transfer_range_fraction.pdf"), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def reasoning_switch_fig():
    fp = "data/qwen_reason_transfer_animals_consider_five7.json"
    if not os.path.exists(fp):
        print("(no reasoning-switch data yet)"); return
    d = json.load(open(fp)); B = d["budgets"]
    pn, pt = d["P_nothink"], d["P_think"]
    native_nothink = pn["NT|top"] - pn["NT|bot"]
    q2r = [pt[f"NT|top|{b}"] - pt[f"NT|bot|{b}"] for b in B]          # nothink-fit  -> think
    r2q = [pn[f"TK{b}|top"] - pn[f"TK{b}|bot"] for b in B]            # think-fit    -> nothink
    nat_think = [pt[f"TK|top|{b}"] - pt[f"TK|bot|{b}"] for b in B]    # native think

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=150)
    ax.set_facecolor(SURF)
    x = np.arange(len(B))
    ax.axhline(0, color=MUTED, lw=1)
    ax.axhline(native_nothink, color=GREY, lw=1.4, ls="--")
    ax.text(len(B) - 1.02, native_nothink + .02, f"native nothink ({native_nothink:+.2f})",
            fontsize=9, color=MUTED, ha="right")
    ax.plot(x, nat_think, "-o", color=GREENED, lw=2.2, ms=7, label="native think (fit = eval)")
    ax.plot(x, q2r, "-o", color=BLUE, lw=2.2, ms=7, label="qwen → reasoning-qwen  (nothink-fit, think eval)")
    ax.plot(x, r2q, "-o", color=ORANGE, lw=2.2, ms=7, label="reasoning-qwen → qwen  (think-fit, nothink eval)")
    for xi, vals, col in [(x, nat_think, GREENED), (x, q2r, BLUE), (x, r2q, ORANGE)]:
        for xx, vv in zip(xi, vals):
            ax.text(xx, vv + (.03 if vv >= 0 else -.06), f"{vv:+.2f}", ha="center", fontsize=8.3, color=col)
    ax.set_xticks(x); ax.set_xticklabels([f"{b}" for b in B])
    ax.set_xlabel("thinking budget (tokens)", fontsize=10.5, color=INK)
    ax.set_ylabel("steering spread   P(y⁺)$_{top}$ − P(y⁺)$_{bot}$", fontsize=10.5, color=INK)
    ax.set_ylim(-0.45, 1.12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Steering across the Qwen3-8B reasoning switch\n"
                 "native fits stay strong; cross-mode transfer decays (and reverses) with budget",
                 fontsize=12, color=INK, loc="left")
    fig.patch.set_facecolor(SURF); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reasoning_switch_transfer.png"), dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(os.path.join(OUT, "reasoning_switch_transfer.pdf"), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    _, results, recs = T.analyze()
    recs = [r for r in recs if not any(np.isnan(x) for x in r[3:13])]
    cross_model_fig(recs)
    transfer_dist_fig(results)
    delta1_sign_fig(results)
    sign_combined_fig(results)      # MAIN: model-pair heatmap + by-nudge×effect sign bars
    range_fraction_fig(results)     # APPENDIX: fraction-of-range steered
    model_pair_fig(results)
    model_pair_sign_fig(results)
    reasoning_switch_fig()
    print(f"wrote {OUT}/  transfer_summary · transfer_logit_dist · "
          "transfer_delta1_sign · transfer_delta1_subsets · reasoning_switch_transfer  (.png/.pdf)")
