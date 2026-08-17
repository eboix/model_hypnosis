"""Analyze the FOUND-extremizer transfer (sec 4 redo) -- δ₂ (top/bottom-100) ONLY.

For each (nudge, effect, source S, target T) using T's measurement of S's candidates:
  delta1 = l_T(s_top) - l_T(s_bot)          -- the found validated pair (2 queries; s_top/s_bot
                                               chosen by best MEASURED-on-source logit)
  delta2 = max - min over top/bottom-100     -- best gap re-optimized on target (200 queries)
Each normalized by the target's own obtainable range = delta2(T->T) (diagonal).

We DELIBERATELY drop the tilt candidates (the old delta3): `_transfer_cands.py` re-samples
tilt independently of `_scatter_extras.py` (the seed-0 draw did not reproduce), so tilt
configs are misaligned between the two files. top/bottom-100 are deterministic (topk_choices)
and byte-identical, so δ₂ is the trustworthy metric.

Flip = the set straddles l=0 on the target (min<0<max), i.e. steering flips the modal answer.
Stats are reported overall and restricted to source-/target-flippable cells.

NOTE: the reasoning targets (Qwen3.x) are measured in nothink (AT_NOTHINK=1) to match the
§3 steering_heatmap protocol; the other targets have no thinking template.

Writes 2 heatmap figures (one per delta_i) + prints the flip-rate table.
Runs on whatever data/transfer_found/*.json exist (partial-friendly).
"""
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("MHYP_FIGDIR", "figures")
import matplotlib as mpl

TAGS = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b", "qwen25_72b", "qwen3_4b",
        "qwen3_8b", "qwen3_14b", "qwen3_32b", "qwen35_9b", "gemma2_9b", "gemma4_12b",
        "llama31_8b", "phi4", "olmo2_7b", "olmo3_7b"]
LB = {t: t for t in TAGS}
NUD = [("animals_consider", "animals"), ("phrasing_L20_O10", "phrasing"),
       ("jsonblob", "JSON"), ("typos", "typos")]
EFFS = [("five7", "5v7"), ("trolley_yn", "trolley"), ("conscious", "consc.")]


def load():
    D = {}
    for f in glob.glob("data/transfer_found/*.json"):
        d = json.load(open(f)); D[d["target"]] = d
    return D


def arr(entry):
    """concat top+bot as a float array (nan for None) -- δ₂ only, tilt dropped."""
    top = entry.get("top", []); bot = entry.get("bot", [])
    a = np.array([np.nan if v is None else v for v in top + bot], float)
    return a, len(top), len(bot)


def cell(D, nu, ef, S, T):
    e = D.get(T, {}).get(nu, {}).get(ef, {}).get(S)
    return arr(e) if e else (None, 0, 0)


def sides(D, nu, ef, S, T):
    """T's measurement of S's top-100 and bottom-100 configs, as separate float arrays."""
    e = D.get(T, {}).get(nu, {}).get(ef, {}).get(S)
    if not e:
        return None, None
    f = lambda xs: np.array([np.nan if v is None else v for v in xs], float)
    return f(e.get("top", [])), f(e.get("bot", []))


def analyze():
    D = load()
    present = [t for t in TAGS if t in D]
    print(f"targets present: {len(present)}/16  {present}")
    results = {}   # (nu,ef) -> dict of 16x16 arrays + flip bools
    recs = []      # flat records for stats
    for nu, _ in NUD:
        for ef, _ in EFFS:
            n = len(TAGS)
            M = {k: np.full((n, n), np.nan) for k in ("d1", "d2")}
            F = {k: np.full((n, n), np.nan) for k in ("f1", "f2")}
            rng = np.full(n, np.nan)          # target obtainable range = d2(T->T)
            src_flip = np.full(n, np.nan); tgt_flip = np.full(n, np.nan)
            # target ranges + flippability from diagonals
            for j, T in enumerate(TAGS):
                aT, nt, nb = cell(D, nu, ef, T, T)
                if aT is None or np.all(np.isnan(aT)):
                    continue
                rng[j] = np.nanmax(aT) - np.nanmin(aT)
                tgt_flip[j] = (np.nanmin(aT) < 0 < np.nanmax(aT))
            for i, S in enumerate(TAGS):
                aS, nt, nb = cell(D, nu, ef, S, S)     # source's own measurement (diagonal)
                if aS is None or np.all(np.isnan(aS)):
                    continue
                src_flip[i] = (np.nanmin(aS) < 0 < np.nanmax(aS))
                stop = int(np.nanargmax(aS)); sbot = int(np.nanargmin(aS))
                for j, T in enumerate(TAGS):
                    aT, ntT, nbT = cell(D, nu, ef, S, T)
                    if aT is None or np.all(np.isnan(aT)) or np.isnan(rng[j]) or rng[j] == 0:
                        continue
                    d1 = aT[stop] - aT[sbot]
                    d2 = np.nanmax(aT) - np.nanmin(aT)
                    M["d1"][i, j], M["d2"][i, j] = d1 / rng[j], d2 / rng[j]
                    F["f1"][i, j] = (aT[stop] > 0) and (aT[sbot] < 0)
                    F["f2"][i, j] = (np.nanmax(aT) > 0) and (np.nanmin(aT) < 0)
                    if i != j:
                        tarr, barr = sides(D, nu, ef, S, T)
                        topflip = float(np.nanmax(tarr) > 0)      # S's top-100 drives T to y+
                        botflip = float(np.nanmin(barr) < 0)      # S's bottom-100 drives T to y-
                        d1_top = float(aT[stop] > 0)              # S's single top prompt -> y+ on T
                        d1_bot = float(aT[sbot] < 0)              # S's single bot prompt -> y- on T
                        recs.append((nu, i, j, src_flip[i], tgt_flip[j],
                                     F["f1"][i, j], F["f2"][i, j],
                                     d1 / rng[j], d2 / rng[j], topflip, botflip,
                                     d1_top, d1_bot))
            results[(nu, ef)] = {"M": M, "F": F, "rng": rng,
                                 "src_flip": src_flip, "tgt_flip": tgt_flip}
    return D, results, recs


def flip_table(recs):
    recs = [r for r in recs if not any(np.isnan(x) for x in r[3:13])]
    if not recs:
        print("no complete records yet"); return
    sf = np.array([r[3] for r in recs], bool); tf = np.array([r[4] for r in recs], bool)
    f = {1: np.array([r[5] for r in recs]), 2: np.array([r[6] for r in recs])}
    d1 = np.array([r[7] for r in recs]); d2 = np.array([r[8] for r in recs])
    topf = np.array([r[9] for r in recs]); botf = np.array([r[10] for r in recs])
    d1top = np.array([r[11] for r in recs]); d1bot = np.array([r[12] for r in recs])

    def row(mask, name):
        m = mask.astype(bool)
        s = "  ".join(f"{100*f[i][m].mean():5.1f}%" for i in (1, 2)) if m.sum() else "   -"
        print(f"  {name:34s} n={m.sum():4d}   {s}")
    print("\nMODE-FLIP RATE by search budget   (delta1=2 queries | delta2=top/bottom-100)")
    print(f"  {'subset':34s} {'':6s}   d1     d2")
    row(np.ones(len(recs), bool), "all off-diagonal pairs")
    row(sf, "source flippable")
    row(tf, "target flippable")
    row(sf & tf, "both flippable")
    for nu, lab in NUD:
        mask = np.array([r[0] == nu for r in recs]) & sf & tf
        row(mask, f"both flippable / {lab}")

    # ---- directional: does the source's top-/bottom-100 flip the target? ----
    def drow(mask, name):
        m = mask.astype(bool)
        if not m.sum():
            print(f"  {name:26s}     -"); return
        both = (topf[m] > 0) & (botf[m] > 0)
        print(f"  {name:26s} n={m.sum():4d}   top-100→y+ {100*topf[m].mean():5.1f}%   "
              f"bot-100→y- {100*botf[m].mean():5.1f}%   both {100*both.mean():5.1f}%")
    print("\nDIRECTIONAL FLIP (top/bottom-100 of source, measured on target; crosses ℓ=0)")
    drow(tf, "target flippable")
    drow(sf & tf, "both flippable")
    for nu, lab in NUD:
        drow(np.array([r[0] == nu for r in recs]) & sf & tf, f"  {lab}")

    # ---- delta1: the single found extremizing PAIR (s_top, s_bot), measured on target ----
    def d1row(mask, name):
        m = mask.astype(bool)
        if not m.sum():
            print(f"  {name:26s}     -"); return
        pos = d1[m] > 0; neg = d1[m] < 0
        print(f"  {name:26s} n={m.sum():4d}   top→y+ {100*d1top[m].mean():5.1f}%   "
              f"bot→y- {100*d1bot[m].mean():5.1f}%   both {100*f[1][m].mean():5.1f}%   "
              f"|  δ1>0 {100*pos.mean():5.1f}%  δ1<0 {100*neg.mean():5.1f}%")
    print("\nDELTA1 FOUND-PAIR (2 queries): direction of ℓ_T(s_top)−ℓ_T(s_bot)")
    d1row(np.ones(len(recs), bool), "all off-diagonal")
    d1row(tf, "target flippable")
    d1row(sf & tf, "both flippable")
    for nu, lab in NUD:
        d1row(np.array([r[0] == nu for r in recs]) & sf & tf, f"  {lab}")

    # ---- average logit-range fraction steered: flippable vs overall ----
    def frow(mask, name):
        m = mask.astype(bool)
        if not m.sum():
            print(f"  {name:26s}     -"); return
        print(f"  {name:26s} n={m.sum():4d}   δ1/range {d1[m].mean():.3f} "
              f"(med {np.median(d1[m]):.3f})   δ2/range {d2[m].mean():.3f} (med {np.median(d2[m]):.3f})")
    print("\nAVG LOGIT-RANGE FRACTION STEERED  (δ/range_T; 1.0 = full target range)")
    frow(np.ones(len(recs), bool), "all off-diagonal")
    frow(tf, "target flippable")
    frow(sf & tf, "both flippable")


def heatmaps(results):
    for key, title in [("d1", "delta1 (found pair, 2 queries)"),
                       ("d2", "delta2 (best of top/bottom-100)")]:
        fig, axs = plt.subplots(len(NUD), len(EFFS), figsize=(11, 13.5), dpi=130)
        cmap = mpl.colormaps["magma"].copy(); cmap.set_bad("#dddddd")
        for r, (nu, ndl) in enumerate(NUD):
            for c, (ef, efl) in enumerate(EFFS):
                ax = axs[r, c]; res = results.get((nu, ef))
                if res is None:
                    ax.axis("off"); continue
                im = ax.imshow(res["M"][key], cmap=cmap, vmin=0, vmax=1.2, aspect="equal")
                ax.set_title(f"{ndl}→{efl}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel("source", fontsize=7)
                if r == len(NUD) - 1:
                    ax.set_xlabel("target", fontsize=7)
        fig.suptitle(f"Transfer {title}, normalized by target's obtainable range  "
                     f"(rows=source, cols=target)", fontsize=12, y=0.995)
        cax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
        fig.colorbar(im, cax=cax, extend="max").set_label(f"{key}/range$_T$", fontsize=9)
        fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.04, wspace=0.1, hspace=0.2)
        os.makedirs(OUT, exist_ok=True)
        fig.savefig(os.path.join(OUT, f"transfer_found_{key}.png"), dpi=150, bbox_inches="tight")
        fig.savefig(os.path.join(OUT, f"transfer_found_{key}.pdf"), bbox_inches="tight")
        plt.close(fig)
    print("wrote figures/transfer_found_{d1,d2}.png/.pdf")


if __name__ == "__main__":
    D, results, recs = analyze()
    flip_table(recs)
    if len(D) >= 8:
        heatmaps(results)
    else:
        print(f"\n(only {len(D)} targets present -- skipping heatmaps until more land)")
