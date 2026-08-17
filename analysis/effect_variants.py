"""Effect-variant generalization: do the SAME items push variant questions the
same way? Correlates per-item betas between an effect and its variants
(trolley vs two rephrasings; redblue vs crimson/aquamarine), per model x pool,
and reports baselines + steering ranges of the variant cells.

    python analysis/effect_variants.py
"""
import glob
import json
import os

import numpy as np

PAIRS = [("trolley", "trolley2"), ("trolley", "trolley3"),
         ("trolley2", "trolley3"), ("redblue", "crimaqua")]


def beta(tag, pool, eff):
    fp = f"data/cells/{tag}/{pool}_{eff}/fit.json"
    if not os.path.exists(fp):
        return None, None
    d = json.load(open(fp))
    return np.array(d["beta"]), d


def main():
    tags = sorted({p.split(os.sep)[2] for p in glob.glob("data/cells/*/animals_trolley2")})
    for pool in ("animals", "foods"):
        print(f"\n===== pool: {pool} =====")
        print(f"{'model':<12}" + "".join(f"{a}~{b:<10}"[:22].ljust(22) for a, b in PAIRS)
              + "  variant baselines (t2, t3, crimaqua)")
        for tag in tags:
            row = f"{tag:<12}"
            for a, b in PAIRS:
                ba, _ = beta(tag, pool, a)
                bb, _ = beta(tag, pool, b)
                row += (f"{np.corrcoef(ba, bb)[0,1]:>8.2f}".ljust(22)
                        if ba is not None and bb is not None else f"{'-':>8}".ljust(22))
            bases = []
            for e in ("trolley2", "trolley3", "crimaqua"):
                _, d = beta(tag, pool, e)
                bases.append(f"{d['baseline_p']:.2f}" if d else "-")
            print(row + "  " + "/".join(bases))
        # steering ranges of variants
        print("  variant steering (bot..top mean P):")
        for tag in tags:
            parts = []
            for e in ("trolley2", "trolley3", "crimaqua"):
                sp = f"data/cells/{tag}/{pool}_{e}/steer.json"
                if os.path.exists(sp):
                    s = json.load(open(sp))
                    sig = lambda v: 1/(1+np.exp(-np.mean([y for _, y in v]))) if v else float("nan")
                    parts.append(f"{e}:{sig(s['bot']):.2f}..{sig(s['top']):.2f}")
            if parts:
                print(f"    {tag:<12} " + "  ".join(parts))


if __name__ == "__main__":
    main()
