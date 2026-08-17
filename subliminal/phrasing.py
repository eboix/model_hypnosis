"""Phrasing nudge: 20 sentence slots x 10 paraphrases each (bank from the
committed data/sentences20x10.json). Additive model over slot choices:

    logit p = mu + sum_g delta[g, choice_g]      (delta sum-to-zero per slot)

topk_choices enumerates the exact K best/worst full choices via best-first
search over per-slot rank vectors (ported from legacy plot_phrasing20_sweep).
"""
import heapq
import json

import numpy as np
try:
    from scipy import sparse
except ImportError:          # vllm containers ship neither scipy nor sklearn
    sparse = None
try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None

from subliminal.fitting import logit


def load_bank(path="data/sentences20x10.json"):
    d = json.load(open(path))
    groups = d["groups"]
    return groups, len(groups), len(groups[0])


def build_prompt(groups, choice, question):
    para = " ".join(groups[g][choice[g]] for g in range(len(groups)))
    return f"{para} {question}"


def fit_groups(trials, NG, O, alpha=3.0):
    """trials = [(choice, l)] with l = exact log-odds -> (mu, delta, r2)."""
    rows, cols, y = [], [], []
    for r, (ch, l) in enumerate(trials):
        for g, o in enumerate(ch):
            rows.append(r)
            cols.append(g * O + o)
        y.append(l)
    y = np.array(y)
    if sparse is not None and Ridge is not None:
        X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=(len(trials), NG * O))
        reg = Ridge(alpha=alpha, solver="lsqr")
        reg.fit(X, y)
        coef, intercept = reg.coef_, float(reg.intercept_)
        pred = reg.predict(X)
    else:
        # dense numpy ridge with unpenalized intercept (matches sklearn Ridge)
        Xd = np.zeros((len(trials), NG * O))
        Xd[rows, cols] = 1.0
        xm, ym = Xd.mean(axis=0), y.mean()
        Xc = Xd - xm
        coef = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(NG * O),
                               Xc.T @ (y - ym))
        intercept = float(ym - xm @ coef)
        pred = Xd @ coef + intercept
    delta = coef.reshape(NG, O)
    mu = float(intercept + delta.mean(axis=1).sum())
    delta = delta - delta.mean(axis=1, keepdims=True)
    r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    return mu, delta, r2


def topk_choices(delta, K, maximize=True, free=None):
    """Exact top-K choice vectors by sum of deltas; slots not in `free` are
    pinned to option 0."""
    NG, O = delta.shape
    free = list(range(NG)) if free is None else list(free)
    sgn = 1.0 if maximize else -1.0
    order = {g: sorted(range(O), key=lambda j: -sgn * delta[g, j]) for g in free}
    sd = {g: [sgn * delta[g, order[g][r]] for r in range(O)] for g in free}
    start = tuple([0] * len(free))
    heap = [(0.0, start)]
    seen = {start}
    out = []
    while heap and len(out) < K:
        deficit, ranks = heapq.heappop(heap)
        ch = [0] * NG
        for gi, g in enumerate(free):
            ch[g] = order[g][ranks[gi]]
        out.append(tuple(ch))
        for gi, g in enumerate(free):
            if ranks[gi] + 1 < O:
                nr = list(ranks)
                nr[gi] += 1
                nr = tuple(nr)
                if nr not in seen:
                    seen.add(nr)
                    heapq.heappush(
                        heap, (deficit + sd[g][ranks[gi]] - sd[g][ranks[gi] + 1], nr))
    return out


def tilted_choices(delta, rng, per_tau=50, step=0.05, taus=None):
    NG, O = delta.shape
    if taus is None:
        taus = np.concatenate([np.linspace(-40, -4, 16), np.linspace(-4, 4, 9),
                               np.linspace(4, 40, 16)])
    bins = {}
    for tau in taus:
        probs = np.exp(tau * delta)
        probs /= probs.sum(axis=1, keepdims=True)
        for _ in range(per_tau):
            ch = tuple(int(rng.choice(O, p=probs[g])) for g in range(NG))
            b = round(sum(delta[g, ch[g]] for g in range(NG)) / step)
            if b not in bins or rng.random() < 0.3:
                bins[b] = ch
    return list(bins.values())


def choice_pred(delta, ch):
    return float(sum(delta[g, o] for g, o in enumerate(ch)))
