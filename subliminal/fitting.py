"""Additive-model fitting and list construction for steering.

Model: logit p = a + sum_{i in list} beta_i  (ridge on the 0/1 item-membership
design; exact per-trial logits from the backend, so no measurement noise).
"""
import itertools
import json
import math

import numpy as np
try:
    from scipy import sparse
    from sklearn.linear_model import Ridge
except ImportError:          # vllm containers ship neither scipy nor sklearn
    sparse = Ridge = None

# Clip only at double-precision resolution: stored p retains logit information
# to ~|30| nats; the old 1e-4 clip flattened everything beyond |9.21| and wrecked
# fits in near-saturated cells (the reads themselves are exact).
EPS = 1e-15
NA = 100  # pool size
L = 10    # items per list in the main collection


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def load_trials(raw_path):
    """[(idx list, p, l)] for valid trials; l = exact log-odds when recorded
    (newer collections), else logit(p) from the double-precision p."""
    out = []
    for line in open(raw_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        p = r.get("p5")
        if p is not None:
            out.append((r["idx"], p, r.get("l", logit(p))))
    return out


def fit_betas(trials, alpha=10.0, n_items=None):
    """Ridge additive fit on exact log-odds -> (a, beta[n_items], per_trial_r2).

    n_items defaults to the vocabulary implied by the trials, so the same call
    works for the 100-, 200- and 1000-animal pools."""
    rows, cols = [], []
    y = []
    for r, (idx, _p, l) in enumerate(trials):
        for i in idx:
            rows.append(r)
            cols.append(i)
        y.append(l)
    NA_ = n_items or (max(cols) + 1 if cols else NA)
    y = np.array(y)
    if sparse is not None:
        X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=(len(trials), NA_))
        reg = Ridge(alpha=alpha, solver="lsqr")
        reg.fit(X, y)
        intercept, coef = float(reg.intercept_), reg.coef_.copy()
        pred = reg.predict(X)
    else:
        # dense numpy ridge with unpenalized intercept (matches sklearn Ridge)
        Xd = np.zeros((len(trials), NA_))
        Xd[rows, cols] = 1.0
        xm, ym = Xd.mean(axis=0), y.mean()
        Xc = Xd - xm
        coef = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(NA_),
                               Xc.T @ (y - ym))
        intercept = float(ym - xm @ coef)
        pred = Xd @ coef + intercept
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return intercept, coef, float(r2)


def binned_r2(trials, a, beta, nbins=40):
    """R^2 of the additive prediction against quantile-binned mean logits.
    Bin count scales down for small n (smoke runs); NaN if too few bins."""
    X = np.array([beta[idx].sum() for idx, _p, _l in trials])
    Yl = np.array([l for _idx, _p, l in trials])
    nbins = max(4, min(nbins, len(trials) // 25))
    qs = np.quantile(X, np.linspace(0, 1, nbins + 1))
    qs[-1] += 1e-9
    b = np.clip(np.digitize(X, qs) - 1, 0, nbins - 1)
    bx, by = [], []
    for k in range(nbins):
        m = b == k
        if m.sum() >= 3:
            bx.append(X[m].mean())
            by.append(Yl[m].mean())
    if len(bx) < 3:
        return float("nan")
    bx, by = np.array(bx), np.array(by)
    sl, ic = np.polyfit(bx, by, 1)
    pred = ic + sl * bx
    return float(1 - np.sum((by - pred) ** 2) / np.sum((by - by.mean()) ** 2))


def extreme_subsets(beta, K, ll=L, maximize=True, slack=4, cap=60000):
    """K highest/lowest predicted-score ll-subsets, drawn exactly from the
    (ll + slack) most extreme items (K << C(ll+slack, ll), so this is exact)."""
    order = np.argsort(beta)
    M = ll + slack
    while M > ll and math.comb(M, ll) > cap:
        M -= 1
    pool = order[-M:] if maximize else order[:M]
    subs = list(itertools.combinations([int(i) for i in pool], ll))
    subs.sort(key=lambda s: sum(beta[i] for i in s), reverse=maximize)
    return subs[:K]


def tilted_lists(beta, rng, per_tau=60, step=0.08,
                 taus=None, ll=L):
    """Lists spanning the predicted-score range: sample items with prob
    proportional to exp(tau*beta), sweep tau, keep ~one list per score bin."""
    if taus is None:
        taus = np.concatenate([np.linspace(-40, -4, 16),
                               np.linspace(-4, 4, 9),
                               np.linspace(4, 40, 16)])
    bins = {}
    for tau in taus:
        w = np.exp(tau * (beta - beta.mean()))
        w /= w.sum()
        for _ in range(per_tau):
            lst = tuple(sorted(int(i) for i in rng.choice(NA, ll, replace=False, p=w)))
            b = round(beta[list(lst)].sum() / step)
            if b not in bins or rng.random() < 0.3:
                bins[b] = lst
    return list(bins.values())
