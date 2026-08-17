"""Shared base for steering the closed-weight API reasoning models (Section 3.2).

This module is the single place that knows how to (a) sample one binary
forced-choice answer from a paid provider+model, and (b) run the additive-fit
steering procedure of Appendix A.4 / Table 1 on top of those samples:

    collect N random configs -> logistic fit -> candidate enumeration
        -> screen K_scr -> confirm K_conf -> report on 100 fresh held-out draws.

Three thin CLIs (``openai_gpt56.py``, ``gemini.py``, ``anthropic.py``) import
these primitives and wire them to per-model reasoning settings and budgets.

Provider endpoints (unchanged from the source scripts):
  * OpenAI       chat.completions (reasoning_effort)         -- GPT-5.6 sol/terra
  * Gemini       Google Generative Language generateContent  -- gemini-2.5/3 flash
  * Anthropic    https://api.anthropic.com/v1/messages        -- Claude Haiku/Sonnet

COST WARNING -- every function here that takes a ``Clients`` hits a PAID
endpoint. Callers must opt in explicitly (the CLIs gate real calls behind
``--confirm-paid`` and ship small default budgets). One sampled answer == one
billed request; ``collect_raw`` and ``report@100`` fan out to thousands.

ALL keys are loaded through ``mhyp.keys.get_key`` (env var, never a home file).
"""
import asyncio
import json
import os
import heapq

import numpy as np

from mhyp import config
from mhyp.keys import get_key

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent?key={key}")

# Bank cue -> (bank file stem under data/, options per slot). Mirrors the source
# saturation scripts (phrasing_L20_O10 uses O10; jsonblob/typos use O6).
BANK_FILES = {
    "phrasing_L20_O10": ("sentences20x10", 10),
    "jsonblob": ("json12x6", 6),
    "typos": ("sentences20_typos", 6),
}


# ============================== sampling ====================================

class Clients:
    """Async context manager holding the provider transports + a shared
    concurrency semaphore. The OpenAI client is built lazily so a Gemini/Claude
    run does not require an OpenAI key. Keys via ``get_key`` only."""

    def __init__(self, conc=8, timeout=180.0):
        self.sem = asyncio.Semaphore(conc)
        self.timeout = timeout
        self.http = None          # httpx.AsyncClient (Gemini + Anthropic REST)
        self._oai = None          # openai.AsyncOpenAI (lazy)

    async def __aenter__(self):
        import httpx
        self.http = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *exc):
        if self.http is not None:
            await self.http.aclose()
        if self._oai is not None:
            await self._oai.close()

    @property
    def oai(self):
        if self._oai is None:
            from openai import AsyncOpenAI
            self._oai = AsyncOpenAI(api_key=get_key("openai"),
                                    timeout=self.timeout, max_retries=3)
        return self._oai


def parse_last(text, tok_a, tok_b):
    """Last-token forced-choice parse: 1 if tok_a appears last, 0 if tok_b,
    else None. Identical to every source producer."""
    a, b = tok_a.strip().lower(), tok_b.strip().lower()
    for w in reversed([w.strip(".,!?:;()*'\"").lower() for w in (text or "").split()]):
        if w == a:
            return 1
        if w == b:
            return 0
    return None


async def _call_openai(cl, model, prompt, effort, max_tokens):
    r = await cl.oai.chat.completions.create(
        model=model, reasoning_effort=effort,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens)
    return r.choices[0].message.content


async def _call_gemini(cl, model, prompt, effort, max_tokens):
    # gemini-2.5-*: integer thinkingBudget; gemini-3+ : thinkingLevel low/high.
    hi = effort != "low"
    if "2.5" in model:
        tc = {"thinkingBudget": 2048 if hi else 512}
    else:
        tc = {"thinkingLevel": "high" if hi else "low"}
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"thinkingConfig": tc, "maxOutputTokens": max_tokens}}
    r = await cl.http.post(GEMINI_URL.format(model=model, key=get_key("gemini")),
                           json=body)
    r.raise_for_status()
    parts = r.json()["candidates"][0].get("content", {}).get("parts", []) or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


async def _call_anthropic(cl, model, prompt, effort, max_tokens):
    h = {"x-api-key": get_key("anthropic"), "anthropic-version": ANTHROPIC_VERSION,
         "content-type": "application/json"}
    if model == "claude-sonnet-5":       # adaptive thinking + effort param
        body = {"model": model, "max_tokens": max_tokens,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
                "messages": [{"role": "user", "content": prompt}]}
    else:                                # haiku-4.5 (pre-4.6): scale think budget
        body = {"model": model, "max_tokens": max_tokens,
                "thinking": {"type": "enabled",
                             "budget_tokens": 1024 if effort == "low" else 4096},
                "messages": [{"role": "user", "content": prompt}]}
    r = await cl.http.post(ANTHROPIC_URL, headers=h, json=body)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json().get("content", [])
                   if b.get("type") == "text")


_CALLERS = {"openai": _call_openai, "gemini": _call_gemini, "anthropic": _call_anthropic}


async def sample_answer(cl, provider, model, prompt, tok_a, tok_b,
                        *, effort="low", max_tokens=4000, retries=3):
    """The one primitive: sample a single binary forced-choice answer.

    Returns 1 (tok_a), 0 (tok_b) or None (unparseable / call failed). One billed
    request per call. Retries with backoff on transport errors.
    """
    fn = _CALLERS[provider]
    async with cl.sem:
        for att in range(retries):
            try:
                text = await fn(cl, model, prompt, effort, max_tokens)
                return parse_last(text, tok_a, tok_b)
            except Exception as e:
                if att == retries - 1:
                    print(f"    err {model} {type(e).__name__} {str(e)[:80]}",
                          flush=True)
                    return None
                await asyncio.sleep(2 * (att + 1))


async def pmean(cl, provider, model, prompt, tok_a, tok_b, k, **kw):
    """P(y+) over ``k`` fresh samples of one prompt -> (p, n_parsed)."""
    ys = await asyncio.gather(*[
        sample_answer(cl, provider, model, prompt, tok_a, tok_b, **kw)
        for _ in range(k)])
    g = [y for y in ys if y is not None]
    return ((sum(g) / len(g)) if g else None), len(g)


# ============================== resources ===================================

def load_pool_by_name(name):
    """Resolve a pool by short name via config.DATA (cwd-independent)."""
    from subliminal.pools import load_pool
    return load_pool(str(config.DATA / "pools" / f"{name}.json"))


def load_bank_by_stem(stem, O):
    """First ``O`` fragments of each slot group for bank stem under data/."""
    groups = json.load(open(config.DATA / f"{stem}.json"))["groups"]
    return [g[:O] for g in groups]


def pool_prompter(pool, question):
    from subliminal.pools import build_prompt
    return lambda ch: build_prompt(pool, list(ch), question)


def bank_prompter(groups, question):
    from subliminal.phrasing import build_prompt
    return lambda ch: build_prompt(groups, list(ch), question)


def make_saturation_prompts(cue, question, n, seed):
    """N random prompts for a saturation cell (Fig 9). Pool cue draws distinct
    length-L item lists; bank cues draw one fragment per slot. Mirrors the
    ``make_prompts`` of the source saturation scripts."""
    import random
    rng = random.Random(seed)
    if cue == "animals_consider":
        pool = load_pool_by_name("animals_consider")
        prm = pool_prompter(pool, question)
        na, L = len(pool["items"]), config.CUES[cue]["L"]
        out = []
        for _ in range(n):
            idx = rng.sample(range(na), L)
            rng.shuffle(idx)
            out.append(prm(idx))
        return out
    stem, O = BANK_FILES[cue]
    groups = load_bank_by_stem(stem, O)
    prm = bank_prompter(groups, question)
    return [prm([rng.randrange(len(g)) for g in groups]) for _ in range(n)]


# ============================== fit math ====================================

def fit_irls(X, y, l2=1.0, iters=60):
    """L2-penalised (intercept unpenalised) logistic IRLS. Returns full beta
    with the intercept as the last entry. Matches the numpy fallback used
    across the source scripts (no sklearn needed in the container)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    b = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ b, -30, 30)))
        W = p * (1 - p) + 1e-9
        g = Xb.T @ (y - p) - l2 * np.r_[b[:-1], 0.0]
        H = (Xb * W[:, None]).T @ Xb + l2 * np.eye(Xb.shape[1])
        H[-1, -1] -= l2
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def auc(scores, y):
    """Mann-Whitney AUC."""
    o = np.argsort(scores)
    r = np.empty(len(scores))
    r[o] = np.arange(1, len(scores) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv_auc(X, y, l2=1.0, folds=5):
    """5-fold cross-validated AUC using ``fit_irls``."""
    fo = np.arange(len(y)) % folds
    aucs = []
    for f in range(folds):
        b = fit_irls(X[fo != f], y[fo != f], l2)
        s = np.hstack([X[fo == f], np.ones((int((fo == f).sum()), 1))]) @ b
        aucs.append(auc(s, y[fo == f]))
    return float(np.nanmean(aucs))


def design_bank(rows, NG, O):
    """Slot-option one-hot design [N, NG*O] from bank rows {choices, y}."""
    X = np.zeros((len(rows), NG * O))
    y = np.array([r["y"] for r in rows], float)
    for i, r in enumerate(rows):
        for g, c in enumerate(r["choices"]):
            X[i, g * O + c] = 1.0
    return X, y


def design_presence(rows, NI):
    """Item-presence design [N, NI] from pool rows {choices, y}."""
    X = np.zeros((len(rows), NI))
    y = np.array([r["y"] for r in rows], float)
    for i, r in enumerate(rows):
        X[i, r["choices"]] = 1.0
    return X, y


def design_ip(rows, NI, LP):
    """Item x position (canonical) design [N, NI*LP] from ordered pool rows."""
    X = np.zeros((len(rows), NI * LP))
    y = np.array([r["y"] for r in rows], float)
    for i, r in enumerate(rows):
        for p, it in enumerate(r["choices"]):
            X[i, it * LP + p] = 1.0
    return X, y


# ============================== candidates ==================================

def kbest_bank(delta, ncand, side):
    """Exact top-``ncand`` (side='top') / bottom bank configs by predicted logit
    via a best-first heap over the sum-separable slot objective. delta[NG, O]."""
    NG, O = delta.shape
    ranks = [sorted(range(O), key=(lambda o, s=s: -delta[s, o]) if side == "top"
                    else (lambda o, s=s: delta[s, o])) for s in range(NG)]

    def cost(s, r):
        return abs(delta[s, ranks[s][0]] - delta[s, ranks[s][r]])

    start = tuple([0] * NG)
    heap = [(0.0, start)]
    seen = {start}
    out = []
    while heap and len(out) < ncand:
        c, rk = heapq.heappop(heap)
        out.append([ranks[s][rk[s]] for s in range(NG)])
        for s in range(NG):
            if rk[s] + 1 < O:
                nr = rk[:s] + (rk[s] + 1,) + rk[s + 1:]
                if nr not in seen:
                    seen.add(nr)
                    heapq.heappush(heap, (c + cost(s, rk[s] + 1) - cost(s, rk[s]), nr))
    return out


def combo_candidates(beta, side, L, mpool, ncand):
    """Top-``ncand`` distinct L-item lists drawn from the ``mpool`` most extreme
    items by beta (item-presence pool model). Returns unordered item-index lists."""
    import itertools
    order = np.argsort(beta)
    M = list(order[::-1][:mpool]) if side == "top" else list(order[:mpool])
    key = (lambda c: -beta[list(c)].sum()) if side == "top" \
        else (lambda c: beta[list(c)].sum())
    combos = sorted(itertools.combinations(M, L), key=key)[:ncand]
    return [[int(i) for i in c] for c in combos]


def murty_kbest(B, L, K, maximize):
    """Murty K-best distinct ORDERED length-L assignments for an item x position
    log-odds matrix B[NI, L]. Returns lists placed[position] = item index."""
    from scipy.optimize import linear_sum_assignment
    NI = B.shape[0]
    C = (-B.T).copy() if maximize else (B.T).copy()

    def solve(forced, banned):
        cost = C.copy()
        for (p, j) in banned:
            cost[p, j] = 1e9
        rows_ = [p for p in range(L) if p not in forced]
        used = set(forced.values())
        cols = [j for j in range(NI) if j not in used]
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
        s, asg, ft, bt = heapq.heappop(heap)
        if asg in seen:
            continue
        seen.add(asg)
        placed = [None] * L
        for p, j in asg:
            placed[p] = int(j)
        out.append(placed)
        forced = dict(ft)
        banned = set(bt)
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


# ======================= collection + validation ============================

async def collect_raw(cl, provider, model, trials, prompt_of, tok_a, tok_b,
                      raw_path, *, effort="low", max_tokens=4000, chunk=150):
    """Resumable one-sample-per-config collection -> raw.jsonl of {choices, y}.
    Skips lines already present. Returns the parsed rows (y not None)."""
    done = sum(1 for _ in open(raw_path)) if os.path.exists(raw_path) else 0
    with open(raw_path, "a") as f:
        i = done
        while i < len(trials):
            ch = trials[i:i + chunk]
            ys = await asyncio.gather(*[
                sample_answer(cl, provider, model, prompt_of(c), tok_a, tok_b,
                              effort=effort, max_tokens=max_tokens) for c in ch])
            for c, y in zip(ch, ys):
                f.write(json.dumps({"choices": list(c), "y": y}) + "\n")
            f.flush()
            i += len(ch)
            print(f"[collect] {i}/{len(trials)} "
                  f"(parse {np.mean([y is not None for y in ys]):.2f})", flush=True)
    rows = [json.loads(l) for l in open(raw_path) if l.strip()]
    return [r for r in rows if r["y"] is not None]


class Checkpoint:
    """jsonl-backed (key -> (p, n)) cache so a screen/validate run resumes."""

    def __init__(self, path):
        self.path = path
        self.d = {}
        if os.path.exists(path):
            for l in open(path):
                if l.strip():
                    r = json.loads(l)
                    self.d[r["k"]] = (r["p"], r["n"])
        self.f = open(path, "a")

    def get(self, key):
        return self.d.get(key)

    def put(self, key, p, n):
        self.d[key] = (p, n)
        self.f.write(json.dumps({"k": key, "p": p, "n": n}) + "\n")
        self.f.flush()


async def screen_confirm_report(cl, provider, model, side, cands, prompt_of,
                                 tok_a, tok_b, ckpt, *, k_scr, k_conf, n_report,
                                 n_conf=2, effort="low", max_tokens=4000):
    """The Appendix A.4 / Table 1 held-out validation for one side.

    screen all ``cands`` @ k_scr -> keep the ``n_conf`` best -> confirm @ k_conf
    -> pick the winner -> ESTIMATE it @ ``n_report`` fresh held-out draws (the
    reported number). Every measurement is checkpointed by ordered config.
    Returns {best_p, choices, p_screen, p_conf, n_report}.
    """
    top = side == "top"

    async def measure(round_, ch, k):
        key = f"{round_}|" + ",".join(map(str, ch))
        hit = ckpt.get(key)
        if hit is not None and hit[0] is not None:
            return hit
        p, n = await pmean(cl, provider, model, prompt_of(ch), tok_a, tok_b, k,
                           effort=effort, max_tokens=max_tokens)
        ckpt.put(key, p, n)
        return p, n

    scr = []
    for ch in cands:
        p, _ = await measure(f"{side}_scr", ch, k_scr)
        if p is not None:
            scr.append((p, ch))
    scr.sort(key=lambda z: z[0], reverse=top)
    print(f"  [{side}] screened {len(scr)}/{len(cands)}; "
          f"best@{k_scr}={scr[0][0]:.3f}", flush=True)

    conf = []
    for _, ch in scr[:n_conf]:
        p, _ = await measure(f"{side}_conf", ch, k_conf)
        if p is not None:
            conf.append((p, ch))
    conf.sort(key=lambda z: z[0], reverse=top)
    p_conf, ch_best = conf[0]

    pe, ne = await measure(f"{side}_report", ch_best, n_report)
    print(f"  [{side}] conf@{k_conf}={p_conf:.3f}  REPORT@{n_report}={pe:.3f} "
          f"(n={ne})", flush=True)
    return {"best_p": pe, "choices": [int(i) for i in ch_best],
            "p_screen": scr[0][0], "p_conf": p_conf, "n_report": ne}


# ==================== high-level shared entry points ========================
# These are the procedures the three provider CLIs call; they hold no
# provider-specific logic beyond the ``provider`` string handed in.

async def saturation_scan(cl, provider, models, cues, effects, ns, out,
                          *, effort="low", max_tokens=4000):
    """Fig 9 producer: base P(y+) + parse rate for every (model, cue, effect)
    cell, ``ns`` random prompts, one sample each. Resumable per cell (skips
    keys already in ``out``)."""
    import zlib
    from subliminal.effects import EFFECTS
    report = json.load(open(out)) if os.path.exists(out) \
        else {"provider": provider, "effort": effort, "NS": ns, "cells": {}}
    for model in models:
        for cue in cues:
            for ef in effects:
                key = f"{model}/{cue}/{ef}"
                if key in report["cells"]:
                    print(f"skip {key}", flush=True)
                    continue
                e = EFFECTS[ef]
                prompts = make_saturation_prompts(cue, e["question"], ns,
                                                  zlib.crc32(key.encode()))
                ys = await asyncio.gather(*[
                    sample_answer(cl, provider, model, p, e["tok_a"], e["tok_b"],
                                  effort=effort, max_tokens=max_tokens)
                    for p in prompts])
                good = [y for y in ys if y is not None]
                n1, n0 = sum(good), len(good) - sum(good)
                base = n1 / (n1 + n0) if (n1 + n0) else None
                report["cells"][key] = {"base": base, "n_parsed": len(good),
                                        "parse": len(good) / ns, "n1": n1, "n0": n0}
                bs = f"{base:.3f}" if base is not None else "None"
                print(f"{key:52s} base={bs} parse={len(good) / ns:.2f}", flush=True)
                json.dump(report, open(out, "w"), indent=1)
    print("done ->", out, flush=True)
    return report


async def steer_cell(cl, provider, model, family, effect, out_dir, budgets, *,
                     pool_name=None, L=10, bank_stem=None, O=None,
                     effort="low", max_tokens=4000, seed=3, collect=True):
    """Fig 11 producer for one cell: collect -> fit -> enumerate -> screen ->
    confirm -> report@100 for top & bottom. ``family`` selects the additive
    model / candidate enumeration:

      bank      slot-option logistic  -> exact k-best per-slot configs
      presence  item-presence logistic-> top-M-item combinations
      ip        item x position ridge -> Murty k-best ordered assignments

    ``budgets`` keys: n, k_cand, k_scr, k_conf, n_report, n_conf, mpool, ip_alpha.
    Writes fit.json in ``out_dir`` and returns it.
    """
    import random
    from subliminal.effects import EFFECTS
    eff = EFFECTS[effect]
    q, ta, tb = eff["question"], eff["tok_a"], eff["tok_b"]
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, "raw.jsonl")
    N = budgets["n"]
    rng = random.Random(seed)

    if family == "bank":
        groups = load_bank_by_stem(bank_stem, O)
        NG = len(groups)
        prompt_of = bank_prompter(groups, q)
        trials = [[rng.randrange(O) for _ in range(NG)] for _ in range(N)]
    else:
        pool = load_pool_by_name(pool_name)
        NI = len(pool["items"])
        prompt_of = pool_prompter(pool, q)
        trials = []
        for _ in range(N):
            idx = rng.sample(range(NI), L)
            rng.shuffle(idx)
            trials.append(idx)

    if collect:
        rows = await collect_raw(cl, provider, model, trials, prompt_of, ta, tb,
                                 raw, effort=effort, max_tokens=max_tokens)
    else:
        rows = [r for r in (json.loads(l) for l in open(raw) if l.strip())
                if r["y"] is not None]

    if family == "bank":
        X, y = design_bank(rows, NG, O)
        l2 = 2.0
        coef = fit_irls(X, y, l2)[:-1].reshape(NG, O)
        delta = coef - coef.mean(axis=1, keepdims=True)
        cands = {s: kbest_bank(delta, budgets["k_cand"], s) for s in ("top", "bot")}
        model_kind = "bank"
    elif family == "presence":
        X, y = design_presence(rows, NI)
        l2 = 1.0
        beta = fit_irls(X, y, l2)[:-1]
        beta = beta - beta.mean()
        cands = {s: combo_candidates(beta, s, L, budgets["mpool"], budgets["k_cand"])
                 for s in ("top", "bot")}
        model_kind = "item_presence"
    else:  # ip
        X, y = design_ip(rows, NI, L)
        l2 = budgets["ip_alpha"]
        B = fit_irls(X, y, l2)[:-1].reshape(NI, L)
        cands = {s: murty_kbest(B, L, budgets["k_cand"], s == "top")
                 for s in ("top", "bot")}
        model_kind = "item_x_position"

    a = float(y.mean())
    ndf = float(cv_auc(X, y, l2))
    print(f"[fit] n={len(rows)} base={a:.3f} cvAUC={ndf:.3f} kind={model_kind}",
          flush=True)

    ckpt = Checkpoint(os.path.join(out_dir, "validate_ckpt.jsonl"))
    out = {"provider": provider, "model": model, "effect": effect,
           "family": family, "model_kind": model_kind, "effort": effort,
           "n_fit": len(rows), "base": a, "auc": ndf, "budgets": budgets}
    for side in ("top", "bot"):
        r = await screen_confirm_report(
            cl, provider, model, side, cands[side], prompt_of, ta, tb, ckpt,
            k_scr=budgets["k_scr"], k_conf=budgets["k_conf"],
            n_report=budgets["n_report"], n_conf=budgets["n_conf"],
            effort=effort, max_tokens=max_tokens)
        if family != "bank":
            names = pool["items"]
            r["items"] = [names[i] for i in r["choices"]]
        r["prompt"] = prompt_of(r["choices"])
        out[side] = r
    json.dump(out, open(os.path.join(out_dir, "fit.json"), "w"), indent=1)
    print(f"SPREAD {model} {family}/{effect}: bot={out['bot']['best_p']:.3f} "
          f"<-> top={out['top']['best_p']:.3f} (base {a:.3f})", flush=True)
    return out


async def measure_prompts(cl, provider, model, tasks, out, *,
                          effort="low", max_tokens=4000, k=100):
    """Held-out measurement of already-selected prompts (Fig 11 confirmation /
    transfer probes). ``tasks`` = list of (name, effect, prompt, selection_p).
    Reports P(y+) over ``k`` fresh samples per prompt. Resumable per sample."""
    from subliminal.effects import EFFECTS
    ckpt = Checkpoint(out + ".ckpt.jsonl")
    res = {}
    for name, effect, prompt, selp in tasks:
        e = EFFECTS[effect]
        p, n = await pmean(cl, provider, model, prompt, e["tok_a"], e["tok_b"], k,
                           effort=effort, max_tokens=max_tokens)
        ckpt.put(name, p, n)
        res[name] = {"p_fresh": p, "n": n, "parse": n / k,
                     "p_selection": selp, "effect": effect}
        pf = f"{p:.3f}" if p is not None else "NA"
        sp = f"{selp:.3f}" if selp is not None else "NA"
        print(f"[{name}] fresh P={pf} (n={n}/{k}) | selection={sp}", flush=True)
        json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, flush=True)
    return res


# ============================== shared CLI ==================================

def cli_main(provider, *, default_models, efforts, default_steer_model,
             default_measure_model, sat_out, measure_out, doc):
    """Build + run the argparse CLI shared by the three provider modules.

    Differs across providers only by the ``provider`` string, the default model
    names and the allowed ``efforts``. Every subcommand is gated behind
    ``--confirm-paid`` (PAID endpoints) with small default budgets.
    """
    import argparse

    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--effort", default="low", choices=efforts)
        p.add_argument("--max-tokens", type=int, default=4000)
        p.add_argument("--conc", type=int, default=8)
        p.add_argument("--confirm-paid", action="store_true",
                       help=f"required to issue billed {provider} calls")

    s = sub.add_parser("saturation", help="Fig 9 base P(y+) grid scan")
    s.add_argument("--models", nargs="+", default=default_models)
    s.add_argument("--cues", nargs="+",
                   default=["animals_consider", "phrasing_L20_O10", "jsonblob", "typos"])
    s.add_argument("--effects", nargs="+", default=["five7", "trolley_yn", "conscious"])
    s.add_argument("--ns", type=int, default=24, help="random prompts per cell")
    s.add_argument("--out", default=sat_out)
    common(s)

    t = sub.add_parser("steer", help="Fig 11 collect->fit->enumerate->validate")
    t.add_argument("--model", default=default_steer_model)
    t.add_argument("--family", required=True, choices=["bank", "presence", "ip"])
    t.add_argument("--effect", required=True)
    t.add_argument("--pool", help="pool name for presence/ip families")
    t.add_argument("--bank", choices=list(BANK_FILES), help="bank cue (bank family)")
    t.add_argument("--L", type=int, default=10, help="list length (pool families)")
    t.add_argument("--n", type=int, default=800, help="random fit configs (small default)")
    t.add_argument("--k-cand", type=int, default=40)
    t.add_argument("--k-scr", type=int, default=48)
    t.add_argument("--k-conf", type=int, default=48)
    t.add_argument("--n-conf", type=int, default=2)
    t.add_argument("--n-report", type=int, default=100)
    t.add_argument("--mpool", type=int, default=14, help="top-M item pool (presence)")
    t.add_argument("--ip-alpha", type=float, default=10.0, help="ridge (ip family)")
    t.add_argument("--seed", type=int, default=3)
    t.add_argument("--no-collect", action="store_true",
                   help="reuse an existing raw.jsonl instead of collecting")
    t.add_argument("--out-dir", default=None)
    common(t)

    m = sub.add_parser("measure", help="Fig 11 held-out validation of saved prompts")
    m.add_argument("--model", default=default_measure_model)
    m.add_argument("--fit", nargs="+", required=True, help="fit.json file(s)")
    m.add_argument("--effect", required=True)
    m.add_argument("--k", type=int, default=100)
    m.add_argument("--out", default=measure_out)
    common(m)

    args = ap.parse_args()

    async def run():
        if not args.confirm_paid:
            raise SystemExit(f"refusing to bill {provider} without --confirm-paid")
        async with Clients(conc=args.conc) as cl:
            if args.cmd == "saturation":
                await saturation_scan(cl, provider, args.models, args.cues,
                                      args.effects, args.ns, args.out,
                                      effort=args.effort, max_tokens=args.max_tokens)
            elif args.cmd == "steer":
                budgets = {"n": args.n, "k_cand": args.k_cand, "k_scr": args.k_scr,
                           "k_conf": args.k_conf, "n_conf": args.n_conf,
                           "n_report": args.n_report, "mpool": args.mpool,
                           "ip_alpha": args.ip_alpha}
                res_name = args.pool if args.family != "bank" else args.bank
                out_dir = args.out_dir or str(
                    config.CELLS / args.model /
                    f"{args.family}_{res_name}_{args.effect}_{args.effort}")
                stem = BANK_FILES[args.bank][0] if args.family == "bank" else None
                O = BANK_FILES[args.bank][1] if args.family == "bank" else None
                await steer_cell(cl, provider, args.model, args.family, args.effect,
                                 out_dir, budgets, pool_name=args.pool, L=args.L,
                                 bank_stem=stem, O=O, effort=args.effort,
                                 max_tokens=args.max_tokens, seed=args.seed,
                                 collect=not args.no_collect)
            elif args.cmd == "measure":
                tasks = []
                for fp in args.fit:
                    d = json.load(open(fp))
                    for side in ("top", "bot"):
                        if side in d and "prompt" in d[side]:
                            tasks.append((f"{fp}:{side}", args.effect,
                                          d[side]["prompt"], d[side].get("best_p")))
                await measure_prompts(cl, provider, args.model, tasks, args.out,
                                      effort=args.effort, max_tokens=args.max_tokens,
                                      k=args.k)

    asyncio.run(run())
