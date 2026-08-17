"""Open-weight REASONING-model steering: the Appendix A.4 fit-then-validate
pipeline for a local reasoning model (Section 3.2).

A reasoning model emits a chain of thought before the answer, so the exact
position-1 forced-logit read used for the non-reasoning cells (mhyp.collect) is
unavailable. Instead we SAMPLE a full generation per prompt at recommended
decoding (T=0.6, top_p=0.95), parse the final answer token to y in {1, 0, None},
and fit the additive model as a LOGISTIC GLM on the same per-slot / per-item x
position features. The "thinking axis" is either a hard Qwen3 token budget
(sample_binary_budget, forced </think> early exit) or a gpt-oss reasoning-effort
level (sample_binary_effort, harmony final channel).

Stages (``--stage``; default ``all`` = collect -> fit -> reval):

  collect  GPU. Sample N random-configuration outcomes at the chosen budget/
           effort, discarding unparseable draws, into ``raw.jsonl`` (resumable,
           seed-deterministic). Schema ``{"i", "choices", "y"}`` -- byte-for-byte
           the schema the downstream fitter reads.
             (ported from data/_think_collect.py + data/_eff_collect.py collection loops)

  fit      CPU. Read raw.jsonl and fit the logistic GLM, then enumerate the
           single best top/bottom candidate. Writes ``fit.json``.
             bank cues (phrasing/json/typos): per-slot logistic -> delta[slot, option].
             pool cues (animals):             item x position logistic -> beta[item, position].
             (ported from the fit blocks of _think_collect.py / _eff_collect.py,
              using the item x position + Murty model that is now canonical, cf.
              data/_reval100.py)

  steer    GPU. Single-round (biased) validation: enumerate the top/bottom
           K_cand candidates, sample K_scr generations of each, report the best
           per side. Writes ``steer.json``. This is the winner's-curse estimate
           that ``reval`` removes.
             (ported from the validate block of _think_collect.py / _eff_collect.py)

  reval    GPU. The unbiased held-out pipeline: enumerate K_cand candidates
           (k-best via topk_choices for banks / Murty's exact K-best assignment
           for pools), SCREEN all with K_scr, CONFIRM the 2 best with N_val fresh
           generations, and REPORT the winner on N_est further FRESH held-out
           generations (disjoint RNG seed ranges -> stages independent). Writes
           ``reval100.json``.
             (ported verbatim from data/_reval100.py)

  budget   GPU. Dose-response of steering vs CoT length: measure the fit's
           top/bottom extremes at a sweep of thinking budgets (forced </think>
           exit -> every trial terminates). Writes ``think_budget.json``.
             (ported from data/_think_budget.py)

Outputs live under ``config.cell_dir(<model>_<arm>, cue, effect)`` where the
reasoning arm is folded into the tag exactly as the sources name it:
``qwen3_8b_think1024``, ``qwen3_8b_nothink``, ``gptoss_20b_low``.

Requires a GPU: LocalModel loads the weights under vLLM.

  python -m mhyp.reasoning --model qwen3_8b --arm think --budget 1024 \
      --cue animals_consider --effect five7 [--n 20000]
  python -m mhyp.reasoning --model gptoss_20b --arm effort --effort low \
      --cue jsonblob --effect five7
"""
import argparse
import glob
import heapq
import json
import os
import random

import numpy as np

from subliminal.effects import EFFECTS
from subliminal.models import resolve
from subliminal.pools import load_pool, build_prompt
from subliminal.phrasing import topk_choices

from mhyp import config

# ---- numerics preserved verbatim from the source scratch producers ----
COLLECT_SEED = 3            # trials RNG seed (_think_collect/_eff_collect gen(seed=3))
CHUNK = 500                 # trials per vLLM batch (TC_CHUNK / EC_CHUNK)
COLLECT_SEED0 = 300_000     # per-request seed base for collection
SCREEN_SEED = 900_000       # screen / single-round-validate seed base
VAL_SEED = 500_000          # confirm (N_val) seed base
EST_SEED = 700_000          # report (N_est) held-out seed base
BANK_L2, BANK_ITERS = 2.0, 60      # per-slot logistic (fit_bank)
IP_L2, IP_ITERS = 5.0, 30          # item x position logistic (fit_ip_pool)
K_CAND = 40                 # candidates enumerated per side (KTOP / RV_KCAND)
K_SCR = 48                  # screen generations per candidate (TC_K / RV_KSCR)
N_VAL = 100                 # confirm generations on each of the top-2 (RV_NVAL)
N_EST = 100                 # held-out report generations on the winner (RV_NEST)
BUDGET_DEFAULTS = [256, 1024, 2048, 4096]   # TB_BUDGETS
BUDGET_K = 48               # samples per (budget, side) in the dose-response (TB_K)
BUDGET_SEED0 = 7000         # TB seed base


# =========================================================================
# cell spec: cue-family geometry + prompt/design builders (from config)
# =========================================================================

class Spec:
    """Resolves one cue into its pool/bank objects and the prompt / feature
    builders, mirroring mhyp.collect / mhyp.fit. `nfeat` is the logistic design
    width (bank: L*O per-slot; pool: NA*L item x position)."""

    def __init__(self, cue, question):
        self.cue = cue
        self.q = question
        self.is_pool = config.is_pool(cue)
        c = config.CUES[cue]
        if self.is_pool:
            self.pool = load_pool(str(config.DATA / "pools" / f"{c['pool']}.json"))
            self.NA = len(self.pool["items"])
            self.L = c["L"]
            self.nfeat = self.NA * self.L
        else:
            bank = json.load(open(config.DATA / f"{c['bank']}.json"))["groups"]
            self.O = c["O"]
            self.NG = c["L"]
            self.groups = [g[:self.O] for g in bank][:self.NG]
            self.nfeat = self.NG * self.O

    # ---- prompt for one ordered configuration ----
    def prompt(self, ch):
        if self.is_pool:
            return build_prompt(self.pool, list(ch), self.q)
        return " ".join(self.groups[g][c] for g, c in enumerate(ch)) + " " + self.q

    # ---- N random configurations (seed-deterministic) ----
    def trials(self, n, seed=COLLECT_SEED):
        rng = random.Random(seed)
        if self.is_pool:
            out = []
            for _ in range(n):
                idx = rng.sample(range(self.NA), self.L)
                rng.shuffle(idx)
                out.append(idx)
            return out
        return [[rng.randrange(self.O) for _ in range(self.NG)] for _ in range(n)]

    # ---- one 0/1 design row for a configuration ----
    def design_row(self, ch, x):
        if self.is_pool:
            for p, it in enumerate(ch):
                x[it * self.L + p] = 1
        else:
            for g, c in enumerate(ch):
                x[g * self.O + c] = 1
        return x


# =========================================================================
# logistic GLM (IRLS/Newton, unpenalized intercept) -- from _reval100.py
# =========================================================================

def _newton(Xb, y, l2, iters):
    """IRLS/Newton logistic fit on design Xb (last column is the intercept,
    which is left unpenalized). Returns the coefficient vector incl. intercept."""
    b = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ b, -30, 30)))
        W = p * (1 - p) + 1e-9
        g = Xb.T @ (y - p) - l2 * np.r_[b[:-1], 0]
        H = (Xb * W[:, None]).T @ Xb + l2 * np.eye(len(b))
        H[-1, -1] -= l2
        b += np.linalg.solve(H, g)
    return b


def _design(spec, rows):
    X = np.zeros((len(rows), spec.nfeat))
    for t, r in enumerate(rows):
        spec.design_row(r["choices"], X[t])
    y = np.array([r["y"] for r in rows], float)
    return X, y


def fit_glm(spec, rows):
    """Fit the cell's logistic GLM. Returns (kind, params) where params is the
    per-slot delta (bank, sum-to-zero per slot) or the item x position matrix
    B[item, position] (pool)."""
    X, y = _design(spec, rows)
    if spec.is_pool:
        Xb = np.hstack([X, np.ones((len(X), 1))])
        b = _newton(Xb, y, IP_L2, IP_ITERS)
        return "item_x_position", b[:-1].reshape(spec.NA, spec.L)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    b = _newton(Xb, y, BANK_L2, BANK_ITERS)
    coef = b[:-1].reshape(spec.NG, spec.O)
    return "per_slot_logistic", coef - coef.mean(1, keepdims=True)


def _auc(scores, y):
    o = np.argsort(scores)
    r = np.empty(len(scores))
    r[o] = np.arange(1, len(scores) + 1)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def cv_auc(spec, rows, folds=5):
    """5-fold cross-validated AUC of the logistic GLM (diagnostic)."""
    X, y = _design(spec, rows)
    if y.min() == y.max():
        return float("nan")
    l2, it = (IP_L2, IP_ITERS) if spec.is_pool else (BANK_L2, BANK_ITERS)
    fold = np.arange(len(y)) % folds
    aucs = []
    for f in range(folds):
        tr, te = fold != f, fold == f
        b = _newton(np.hstack([X[tr], np.ones((tr.sum(), 1))]), y[tr], l2, it)
        s = np.hstack([X[te], np.ones((te.sum(), 1))]) @ b
        aucs.append(_auc(s, y[te]))
    return float(np.mean(aucs))


# =========================================================================
# candidate enumeration: bank top-K (best-first) / pool Murty K-best
# =========================================================================

def kbest_assignments(B, L, K, maximize):
    """Murty's exact K-best assignment, positions(L) x items(NA). B[item, pos].
    Each position gets a distinct item; returns placed[pos]=item, best first.
    (verbatim from _reval100.py)"""
    from scipy.optimize import linear_sum_assignment
    NIi = B.shape[0]
    C = (-B.T).copy() if maximize else (B.T).copy()   # rows=positions, cols=items

    def solve(forced, banned):
        cost = C.copy()
        for (p, j) in banned:
            cost[p, j] = 1e9
        rows_ = [p for p in range(L) if p not in forced]
        used = set(forced.values())
        cols = [j for j in range(NIi) if j not in used]
        r, c = linear_sum_assignment(cost[np.ix_(rows_, cols)])
        a = dict(forced)
        s = sum(C[p, forced[p]] for p in forced)
        for ri, ci in zip(r, c):
            j = cols[ci]
            if cost[rows_[ri], j] >= 1e8:
                return None
            a[rows_[ri]] = j
            s += C[rows_[ri], j]
        return s, tuple(sorted(a.items()))

    first = solve({}, frozenset())
    if not first:
        return []
    heap = [(first[0], first[1], (), ())]
    out, seen = [], set()
    while heap and len(out) < K:
        s, asg, forced_t, banned_t = heapq.heappop(heap)
        if asg in seen:
            continue
        seen.add(asg)
        placed = [None] * L
        for p, j in asg:
            placed[p] = int(j)
        out.append(placed)
        forced = dict(forced_t)
        banned = set(banned_t)
        f2 = dict(forced)
        for (p, j) in asg:
            if p in forced:
                continue
            nb = frozenset(banned | {(p, j)})
            child = solve(f2, nb)
            if child:
                heapq.heappush(heap, (child[0], child[1],
                                      tuple(sorted(f2.items())), tuple(sorted(nb))))
            f2[p] = j
    return out


def enumerate_candidates(spec, kind, params, k, top):
    """The top- (top=True) or bottom- (top=False) k highest-scoring configs."""
    if spec.is_pool:
        return kbest_assignments(params, spec.L, k, top)
    return [list(c) for c in topk_choices(params, k, top)]


def score(spec, kind, params, ch):
    if spec.is_pool:
        return float(sum(params[it, p] for p, it in enumerate(ch)))
    return float(sum(params[g, c] for g, c in enumerate(ch)))


# =========================================================================
# reasoning-arm sampling wrapper
# =========================================================================

class Arm:
    """The thinking axis: a Qwen token budget, a gpt-oss reasoning-effort level,
    or a plain/nothink chat-template toggle. `.sample` returns (ys, toks, capped);
    toks/capped are None for the non-budget arms."""

    def __init__(self, arm, budget=None, effort=None, markers="qwen",
                 answer_tokens=512):
        self.arm = arm
        self.budget = budget
        self.effort = effort
        self.markers = markers
        self.answer_tokens = answer_tokens

    @property
    def suffix(self):
        if self.arm == "think":
            return f"think{self.budget}"
        if self.arm == "effort":
            return self.effort           # e.g. "low" -> gptoss_20b_low
        return self.arm                  # "nothink" / "plain"

    @property
    def is_effort(self):
        return self.arm == "effort"

    def sample(self, m, prompts, ta, tb, seed0):
        if self.arm == "think":
            return m.sample_binary_budget(prompts, ta, tb, budget=self.budget,
                                          answer_tokens=self.answer_tokens,
                                          seed0=seed0, markers=self.markers)
        if self.arm == "effort":
            ys, cats, ntoks = m.sample_binary_effort(prompts, ta, tb, self.effort,
                                                     seed0=seed0)
            return ys, ntoks, [c == "truncated" for c in cats]
        thinking = False if self.arm == "nothink" else None   # plain => None (R1)
        ys = m.sample_binary_batch(prompts, ta, tb, thinking=thinking, seed0=seed0)
        return ys, None, None


def pmean(arm, m, spec, ta, tb, ch, n, seed0):
    """Mean parsed outcome over n sampled generations of one configuration
    (None -> discarded); returns None if none parse."""
    ys, _, _ = arm.sample(m, [spec.prompt(ch)] * n, ta, tb, seed0)
    v = [x for x in ys if x is not None]
    return float(np.mean(v)) if v else None


# =========================================================================
# model / cell helpers
# =========================================================================

def load_model(model, dtype):
    mid, tp = resolve(model)
    from subliminal.backend import LocalModel
    return LocalModel(mid, tp=tp, dtype=dtype, max_model_len=8192)


def read_rows(raw):
    """Parsed (y is not None) records from raw.jsonl."""
    return [r for r in map(json.loads, open(raw)) if r.get("y") is not None]


# =========================================================================
# stages
# =========================================================================

def stage_collect(m, arm, spec, ta, tb, n, cell):
    """Sample N random-configuration outcomes -> raw.jsonl (resumable)."""
    raw = cell / "raw.jsonl"
    trials = spec.trials(n)
    done = sum(1 for _ in open(raw)) if raw.exists() else 0
    print(f"collect {spec.cue}_{ta}/{tb} arm={arm.suffix} N={n}; resume {done}",
          flush=True)
    with open(raw, "a") as f:
        for j in range(done, n, CHUNK):
            idxs = list(range(j, min(j + CHUNK, n)))
            prompts = [spec.prompt(trials[i]) for i in idxs]
            ys, _, _ = arm.sample(m, prompts, ta, tb, COLLECT_SEED0 + idxs[0])
            for i, y in zip(idxs, ys):
                f.write(json.dumps({"i": i, "choices": trials[i], "y": y}) + "\n")
            f.flush()
            print(f"  {idxs[-1] + 1}/{n} parse "
                  f"{np.mean([y is not None for y in ys]):.2f}", flush=True)
    print(f"wrote {raw}", flush=True)


def stage_fit(arm, spec, cue, effect, model, cell, k_cand):
    """Fit the logistic GLM and enumerate the single best top/bot config -> fit.json."""
    rows = read_rows(cell / "raw.jsonl")
    y = np.array([r["y"] for r in rows], float)
    kind, params = fit_glm(spec, rows)
    auc = cv_auc(spec, rows)
    out = {"tag": f"{model}_{arm.suffix}", "nudge": cue, "eff": effect,
           "arm": arm.arm, "setting": arm.suffix, "n": len(y),
           "base": float(y.mean()), "auc": auc, "model": kind}
    if spec.is_pool:
        out["beta_ip"] = params.tolist()
    else:
        out["delta"] = params.tolist()
    for side, top in (("top", True), ("bot", False)):
        best = enumerate_candidates(spec, kind, params, k_cand, top)[0]
        out[side] = {"choices": [int(c) for c in best],
                     "pred": score(spec, kind, params, best),
                     "para": spec.prompt(best)}
    json.dump(out, open(cell / "fit.json", "w"), indent=1)
    print(f"[fit {cue}/{effect}/{arm.suffix}] n={len(y)} base={y.mean():.3f} "
          f"AUC={auc:.3f} ({kind})", flush=True)


def stage_steer(m, arm, spec, ta, tb, cell, k_cand, k_scr):
    """Single-round (biased) validation of the top/bot K_cand extremes -> steer.json."""
    rows = read_rows(cell / "raw.jsonl")
    kind, params = fit_glm(spec, rows)
    out = {"model": kind, "k_cand": k_cand, "k_scr": k_scr}
    for side, top in (("top", True), ("bot", False)):
        agg = max if top else min
        cfgs = enumerate_candidates(spec, kind, params, k_cand, top)
        res = [(pmean(arm, m, spec, ta, tb, ch, k_scr, SCREEN_SEED), ch)
               for ch in cfgs]
        res = [(p, ch) for p, ch in res if p is not None]
        best = agg(res, key=lambda z: z[0])
        out[side] = {"best_p": best[0],
                     "meanK_p": float(np.mean([p for p, _ in res])),
                     "choices": [int(c) for c in best[1]]}
        print(f"[steer {side}] best_p={best[0]:.3f} "
              f"meanK={out[side]['meanK_p']:.3f}", flush=True)
    json.dump(out, open(cell / "steer.json", "w"), indent=1)
    print(f"SPREAD {arm.suffix}: {out['bot']['best_p']:.3f} <-> "
          f"{out['top']['best_p']:.3f}", flush=True)


def stage_reval(m, arm, spec, cue, effect, ta, tb, model, cell,
                k_cand, k_scr, n_val, n_est):
    """Held-out screen -> confirm -> report on FRESH samples -> reval100.json.
    (verbatim procedure of _reval100.py)"""
    rows = read_rows(cell / "raw.jsonl")
    y = np.array([r["y"] for r in rows], float)
    base = float(y.mean())
    out = {"tag": f"{model}_{arm.suffix}", "nudge": cue, "eff": effect,
           "setting": arm.suffix, "n": len(y), "base": base,
           "k_cand": k_cand, "k_scr": k_scr, "n_val": n_val, "n_est": n_est,
           "model": ("item_x_position" if spec.is_pool else "per_slot_logistic")}
    if y.min() == y.max():                      # one class only -> fit undefined
        out["degenerate"] = True
        out["top"] = out["bot"] = {"best_p": base, "choices": None,
                                   "p_screen": base, "p_val": base}
        json.dump(out, open(cell / "reval100.json", "w"), indent=1)
        print(f"DEGENERATE {cue}/{effect}/{arm.suffix}: base={base:.3f}",
              flush=True)
        return
    out["degenerate"] = False
    kind, params = fit_glm(spec, rows)

    for top in (True, False):
        side = "top" if top else "bot"
        cfgs = enumerate_candidates(spec, kind, params, k_cand, top)
        scr = [(pmean(arm, m, spec, ta, tb, ch, k_scr, SCREEN_SEED), ch)
               for ch in cfgs]
        scr = [(p, ch) for p, ch in scr if p is not None]
        scr.sort(key=lambda z: z[0], reverse=top)          # best-screened first
        top2 = scr[:2]
        val = [(pmean(arm, m, spec, ta, tb, ch, n_val, VAL_SEED), ch, ps)
               for ps, ch in top2]
        val = [v for v in val if v[0] is not None]
        val.sort(key=lambda z: z[0], reverse=top)          # winner = best of top-2
        pv_best, ch_best, ps_best = val[0]
        p_est = pmean(arm, m, spec, ta, tb, ch_best, n_est, EST_SEED)
        out[side] = {"best_p": p_est, "choices": [int(c) for c in ch_best],
                     "p_screen": ps_best, "p_val": pv_best, "n_est": n_est}
        print(f"[{side}] screen{k_scr}={ps_best:.3f} val{n_val}={pv_best:.3f} "
              f"EST{n_est}={p_est:.3f} (fresh held-out, reported)", flush=True)
        json.dump(out, open(cell / "reval100.json", "w"), indent=1)      # per-side checkpoint
    print(f"SPREAD {cue}/{effect}/{arm.suffix}: {out['bot']['best_p']:.3f} <-> "
          f"{out['top']['best_p']:.3f} (base {base:.3f})", flush=True)


def stage_budget(m, arm, spec, ta, tb, model, cell, budgets, budget_k):
    """Dose-response of the fit's top/bot extremes across thinking budgets ->
    think_budget.json. (ported from _think_budget.py)"""
    d = json.load(open(cell / "fit.json"))
    out = {"tag": f"{model}_{arm.suffix}", "eff": d["eff"], "K": budget_k,
           "budgets": {}}
    for B in budgets:
        row = {}
        for side in ("top", "bot"):
            prompt = spec.prompt(d[side]["choices"])
            ys, toks, capped = m.sample_binary_budget(
                [prompt] * budget_k, ta, tb, budget=B, seed0=BUDGET_SEED0,
                markers=arm.markers)
            v = [x for x in ys if x is not None]
            row[side] = {"p": float(np.mean(v)) if v else None, "n": len(v),
                         "forced_exit": float(np.mean(capped)),
                         "toks_mean": float(np.mean(toks)),
                         "toks_med": float(np.median(toks))}
            print(f"[B={B}/{side}] P({ta})={row[side]['p']} n={len(v)}/{budget_k} "
                  f"forced={row[side]['forced_exit']:.2f} "
                  f"toks_med={row[side]['toks_med']:.0f}", flush=True)
        out["budgets"][str(B)] = row
    path = cell / "think_budget.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}", flush=True)


# =========================================================================
# CLI
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="model tag (subliminal.models)")
    ap.add_argument("--cue", required=True, choices=list(config.CUES))
    ap.add_argument("--effect", required=True, choices=list(EFFECTS))
    ap.add_argument("--arm", default="think",
                    choices=["think", "nothink", "plain", "effort"],
                    help="thinking axis: think=Qwen token budget, effort=gpt-oss "
                         "reasoning-effort, nothink/plain=chat-template toggle")
    ap.add_argument("--budget", type=int, default=None,
                    help="Qwen thinking token budget (required for --arm think)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high"],
                    help="gpt-oss reasoning effort (required for --arm effort)")
    ap.add_argument("--markers", default="qwen", choices=["qwen", "gemma"],
                    help="budget close-tag/exit family (sample_binary_budget)")
    ap.add_argument("--answer-tokens", type=int, default=512,
                    help="budget phase-2 answer cap (sample_binary_budget)")
    ap.add_argument("--n", type=int, default=config.N_RANDOM_REASONING,
                    help="random configurations to collect (default 20000)")
    ap.add_argument("--stage", default="all",
                    choices=["all", "collect", "fit", "steer", "reval", "budget"])
    ap.add_argument("--k-cand", type=int, default=K_CAND)
    ap.add_argument("--k-scr", type=int, default=K_SCR)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--n-est", type=int, default=N_EST)
    ap.add_argument("--budgets", default=" ".join(map(str, BUDGET_DEFAULTS)),
                    help="space-separated budgets for --stage budget")
    ap.add_argument("--budget-k", type=int, default=BUDGET_K)
    ap.add_argument("--dtype", default=None,
                    help="override vLLM dtype (default bfloat16; auto for gpt-oss)")
    args = ap.parse_args()

    if args.arm == "think" and args.budget is None:
        ap.error("--arm think requires --budget")
    if args.arm == "effort" and args.effort is None:
        ap.error("--arm effort requires --effort")

    arm = Arm(args.arm, budget=args.budget, effort=args.effort,
              markers=args.markers, answer_tokens=args.answer_tokens)
    eff = EFFECTS[args.effect]
    q, ta, tb = eff["question"], eff["tok_a"], eff["tok_b"]
    spec = Spec(args.cue, q)

    tag = f"{args.model}_{arm.suffix}"
    cell = config.cell_dir(tag, args.cue, args.effect)
    cell.mkdir(parents=True, exist_ok=True)
    dtype = args.dtype or ("auto" if arm.is_effort else "bfloat16")
    budgets = [int(x) for x in args.budgets.split()]

    stages = (["collect", "fit", "reval"] if args.stage == "all"
              else [args.stage])
    need_gpu = any(s in ("collect", "steer", "reval", "budget") for s in stages)
    print(f"=== reasoning {tag} {args.cue}_{args.effect} stages={stages} ===",
          flush=True)

    m = load_model(args.model, dtype) if need_gpu else None
    for s in stages:
        if s == "collect":
            stage_collect(m, arm, spec, ta, tb, args.n, cell)
        elif s == "fit":
            stage_fit(arm, spec, args.cue, args.effect, args.model, cell, args.k_cand)
        elif s == "steer":
            stage_steer(m, arm, spec, ta, tb, cell, args.k_cand, args.k_scr)
        elif s == "reval":
            stage_reval(m, arm, spec, args.cue, args.effect, ta, tb, args.model,
                        cell, args.k_cand, args.k_scr, args.n_val, args.n_est)
        elif s == "budget":
            stage_budget(m, arm, spec, ta, tb, args.model, cell, budgets, args.budget_k)
    print(f"=== reasoning {tag} DONE ===", flush=True)


if __name__ == "__main__":
    main()
