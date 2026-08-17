"""Exp 4: robustness of the animal nudge to frame paraphrase and reordering.

Panel A: per frame paraphrase, mean P(5) of the top-30 / bottom-30 optimized
lists and of 100 shared random lists — does steering survive rewording the
carrier sentence? Panel B: per-list spread of logit P(5) across 8 reorderings
vs the top-bottom gap — is order noise small relative to the effect?

    python analysis/robustness_plot.py --out figures/32_robustness.png
"""
import argparse
import glob
import json
import os

import numpy as np

OUT = os.environ.get("MHYP_FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "32_robustness.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")

    files = sorted(glob.glob("data/cells/*/animals_five7/robustness.json"))
    files = [f for f in files if "_smoke" not in f]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    reorder_rows = []
    for k, fp in enumerate(files):
        tag = fp.split(os.sep)[2]
        d = json.load(open(fp))
        col = cmap(k % 10)
        fids = sorted(d["frames"], key=int)
        tops = [np.mean([y for _, y in d["frames"][fi]["top"]]) for fi in fids]
        bots = [np.mean([y for _, y in d["frames"][fi]["bot"]]) for fi in fids]
        rnds = [np.mean([y for _, y in d["frames"][fi]["random"]]) for fi in fids]
        x = np.arange(len(fids))
        axA.plot(x, sig(tops), "-o", ms=4, color=col, label=tag)
        axA.plot(x, sig(bots), "--o", ms=4, color=col)
        axA.plot(x, sig(rnds), ":", lw=1, color=col)
        gap = np.mean(tops) - np.mean(bots)
        sds = [np.std(r["logits"]) for r in d["reorder"] if len(r["logits"]) > 2]
        if sds:
            reorder_rows.append((tag, gap, float(np.mean(sds)), col))

    axA.set_xlabel("frame paraphrase index (0 = original wording)")
    axA.set_ylabel("P(5)")
    axA.set_ylim(-0.02, 1.02)
    axA.set_title("Steering vs frame paraphrase\n(top-30 solid, bottom-30 dashed, "
                  "random dotted)")
    axA.legend(fontsize=7)

    for tag, gap, sd, col in reorder_rows:
        axB.scatter(gap, sd, s=48, color=col, label=tag)
    if reorder_rows:
        m = max(g for _, g, _, _ in reorder_rows)
        axB.plot([0, m], [0, m], "--", c="gray", lw=1, label="sd = gap")
    axB.set_xlabel("top-bottom steering gap (log-odds)")
    axB.set_ylabel("per-list sd across 8 reorderings (log-odds)")
    axB.set_title("Reordering noise vs steering effect")
    axB.legend(fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print("wrote", args.out)
    print(f"{'model':<12} {'gap(logit)':>10} {'reorder sd':>10}  ratio")
    for tag, gap, sd, _ in reorder_rows:
        print(f"{tag:<12} {gap:>10.2f} {sd:>10.2f}  {sd/max(gap,1e-9):>5.1%}")


if __name__ == "__main__":
    main()
