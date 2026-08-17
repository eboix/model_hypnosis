"""Build data/explorer.json for the interactive model-hypnosis explorer artifact.

Two tiers of cells:
  A. exact-logit steering  (open-weight models x {animals,phrasing,jsonblob,typos} x {five7,
     trolley_yn,conscious}) -- full record: top/bottom selected prompt (per-slot chips shaded
     by share of the predicted gap), logit gap dl, probability gap dp, inverse-Simpson L_eff +
     per-slot shares, subsampled scatter (random cloud + tilt band + top/bottom-100), R^2.
  B. reasoning validated   (qwen3_8b think 256/1024/4096, gpt-oss-20b low) -- prompts, base &
     top/bottom validated probabilities, probability gap, and the number of held-out samples
     used to estimate them (reval100.json n_est). Logit gap is left null (measured by sampling,
     not an exact read).
"""
import os, json
import numpy as np
from subliminal.pools import load_pool
from subliminal.effects import EFFECTS

LB = {"qwen25_05b": "Qwen2.5-0.5B", "qwen25_15b": "Qwen2.5-1.5B", "qwen25_3b": "Qwen2.5-3B",
      "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B", "qwen25_32b": "Qwen2.5-32B",
      "qwen25_72b": "Qwen2.5-72B", "qwen3_4b": "Qwen3-4B", "qwen3_8b": "Qwen3-8B",
      "qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B", "qwen35_9b": "Qwen3.5-9B",
      "gemma2_9b": "Gemma-2-9B", "gemma4_12b": "Gemma-4-12B", "llama31_8b": "Llama-3.1-8B",
      "phi4": "Phi-4", "olmo2_7b": "OLMo-2-7B", "olmo3_7b": "OLMo-3-7B", "mistral7b": "Mistral-7B",
      "gptoss_20b": "gpt-oss-20B"}
# reasoning nudge key -> (bank file, options-per-slot) or None for the animals pool
RBANK = {"phrasing20": ("data/sentences20x10.json", 10), "jsonblob12": ("data/json12x6.json", 6),
         "typos": ("data/sentences20_typos.json", 6)}
ORDER = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b", "qwen25_72b", "qwen3_4b",
         "qwen3_8b", "qwen3_14b", "qwen3_32b", "qwen35_9b", "gemma2_9b", "gemma4_12b",
         "llama31_8b", "phi4", "olmo2_7b", "olmo3_7b"]
NUDGES = [("animals_consider", "animals", "#2a9d3f"), ("phrasing_L20_O10", "phrasing", "#255ea6"),
          ("jsonblob", "JSON", "#e8790c"), ("typos", "typos", "#8e44ad")]
EFFS = [("five7", "5 vs 7"), ("trolley_yn", "trolley"), ("conscious", "conscious")]
BANKS = {"jsonblob": ("data/json12x6.json", 6), "phrasing_L20_O10": ("data/sentences20x10.json", 10),
         "typos": ("data/sentences20_typos.json", 6)}
NLABEL = {n: l for n, l, _ in NUDGES}
RANDN = 500          # subsampled random-cloud points per cell
sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
r2f = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 3)
rnd = lambda x, k=2: None if x is None or not np.isfinite(x) else round(float(x), k)


def neff(g):
    g = np.clip(np.asarray(g, float), 0, None); s = g.sum()
    return float(s * s / np.sum(g * g)) if s > 0 else float("nan")


def selected(tag, nudge, eff, top_ch, bot_ch):
    """(intro, top_frags, bot_frags, tail, g, kind) for the GIVEN top/bottom configs
    (the measured-best choices). g[i] is slot i's contribution to the predicted gap
    between those two prompts (clipped >=0 downstream)."""
    cp = f"data/cells/{tag}/{nudge}_{eff}"; q = EFFECTS[eff]["question"]
    if nudge == "animals_consider":
        fp = f"{cp}/fit_ip.json"
        if not os.path.exists(fp):
            return None
        f = json.load(open(fp)); B = np.array(f["beta_ip"])
        items = load_pool("animals_consider")["items"]
        g = np.array([B[p, top_ch[p]] - B[p, bot_ch[p]] for p in range(len(top_ch))])
        return ("Consider these animals:", [items[i] for i in top_ch],
                [items[i] for i in bot_ch], q, g, "list")
    fp = f"{cp}/fit.json"
    if not os.path.exists(fp) or nudge not in BANKS:
        return None
    bankf, O = BANKS[nudge]
    groups = [gg[:O] for gg in json.load(open(bankf))["groups"]]
    d = json.load(open(fp)); delta = np.array(d["delta"])
    NG = int(d.get("slots", len(groups))); groups = groups[:NG]
    g = np.array([delta[s, top_ch[s]] - delta[s, bot_ch[s]] for s in range(NG)])
    return ("", [groups[s][top_ch[s]] for s in range(NG)],
            [groups[s][bot_ch[s]] for s in range(NG)], q, g, "bank")


def measured_best(cp, nudge, ex):
    """Choices of the MEASURED-best config in each direction, aligned with scatter_extras:
    argmax over top-100 measurements, argmin over bottom-100.  Returns (top_ch, bot_ch,
    tmax, bmin) or None. Animals use se_ip_configs (chs); banks use transfer_cands."""
    top_meas = ex.get("top", {}).get("meas", []); bot_meas = ex.get("bot", {}).get("meas", [])
    if nudge == "animals_consider":
        cfp = f"{cp}/se_ip_configs.json"
        if not os.path.exists(cfp):
            return None
        cf = json.load(open(cfp)); top_chs, bot_chs = cf["top"]["chs"], cf["bot"]["chs"]
    else:
        cfp = f"{cp}/transfer_cands.json"
        if not os.path.exists(cfp):
            return None
        cc = json.load(open(cfp)); top_chs, bot_chs = cc["top"], cc["bot"]
    tvi = [(m, i) for i, m in enumerate(top_meas) if m is not None]
    bvi = [(m, i) for i, m in enumerate(bot_meas) if m is not None]
    if not tvi or not bvi or len(top_chs) < len(top_meas) or len(bot_chs) < len(bot_meas):
        return None
    tmax, ti = max(tvi); bmin, bi = min(bvi)
    return top_chs[ti], bot_chs[bi], tmax, bmin


def random_pred_meas(cellp, fit):
    is_bank = "delta" in fit
    ipp = os.path.join(cellp, "fit_ip.json")
    is_ip = (not is_bank) and os.path.exists(ipp)
    if is_bank:
        mu = float(fit["mu"]); delta = np.array(fit["delta"])
    elif is_ip:
        fi = json.load(open(ipp)); a = float(fi["a"]); B = np.array(fi["beta_ip"])
    else:
        a = float(fit["a"]); beta = np.array(fit["beta"])
    cap = fit.get("n", fit.get("N")); pr, me, chs = [], [], []
    for line in open(os.path.join(cellp, "raw.jsonl")):
        r = json.loads(line)
        if "l" not in r:
            continue
        ch = r.get("ch", r.get("idx"))
        if ch is None:
            continue
        if is_bank:
            pr.append(mu + float(sum(delta[s][c] for s, c in enumerate(ch))))
        elif is_ip:
            pr.append(a + float(sum(B[p, i] for p, i in enumerate(ch))))
        else:
            pr.append(a + float(beta[list(ch)].sum()))
        me.append(r["l"]); chs.append(list(ch))
        if cap and len(pr) >= cap:
            break
    return np.array(pr), np.array(me), chs


def side_choices(cp, nudge):
    """Per-candidate choices aligned with scatter_extras top/bot/tilt (for hover-to-prompt)."""
    try:
        if nudge == "animals_consider":
            cf = json.load(open(f"{cp}/se_ip_configs.json"))
            return {"top": cf["top"]["chs"], "bot": cf["bot"]["chs"],
                    "tilt": cf.get("tilt", {}).get("chs")}
        cc = json.load(open(f"{cp}/transfer_cands.json"))
        return {"top": cc.get("top"), "bot": cc.get("bot"), "tilt": cc.get("tilt")}
    except Exception:
        return {}


def exact_cell(tag, nudge, eff):
    cp = f"data/cells/{tag}/{nudge}_{eff}"
    exf = f"{cp}/scatter_extras.json"; fitf = f"{cp}/fit.json"
    if not (os.path.exists(exf) and os.path.exists(fitf)):
        return None
    ex = json.load(open(exf))
    if nudge == "animals_consider" and ex.get("model") != "item_x_position":
        return None
    mb = measured_best(cp, nudge, ex)
    if mb is None:
        return None
    top_ch, bot_ch, tmax, bmin = mb
    sel = selected(tag, nudge, eff, top_ch, bot_ch)
    if sel is None:
        return None
    intro, tfr, bfr, tail, g, kind = sel
    if g.sum() <= 0:
        return None
    share = (np.clip(g, 0, None) / np.clip(g, 0, None).sum()).tolist()
    fit = json.load(open(fitf))
    rpred, rmeas, rchs = random_pred_meas(cp, fit)
    if len(rpred) == 0:
        return None
    # scatter subsample (points + the choices behind each, so hover can rebuild the prompt)
    idx = np.random.default_rng(0).choice(len(rpred), min(RANDN, len(rpred)), replace=False)
    rand = [[rnd(rpred[i]), rnd(rmeas[i])] for i in idx]
    rc = [rchs[i] for i in idx]
    sc = side_choices(cp, nudge)
    def pts(side):
        s = ex.get(side, {}); pp = s.get("pred", []); mm = s.get("meas", []); ch = sc.get(side) or []
        P, C = [], []
        for i, (p, m) in enumerate(zip(pp, mm)):
            if m is None:
                continue
            P.append([rnd(p), rnd(m)]); C.append(ch[i] if i < len(ch) else None)
        return P, C
    topP, topC = pts("top"); botP, botC = pts("bot")
    t = ex.get("tilt", {}); tch = sc.get("tilt")
    tilt, tiltC = [], []
    for i, (p, m, tau) in enumerate(zip(t.get("pred", []), t.get("meas", []), t.get("tau", []))):
        if m is None:
            continue
        tilt.append([rnd(p), rnd(m), rnd(tau, 3)]); tiltC.append(tch[i] if (tch and i < len(tch)) else None)
    dl = tmax - bmin
    p_top, p_bot = sig(tmax), sig(bmin)
    p_base = sig(float(np.median(rmeas)))
    ipp = f"{cp}/fit_ip.json"
    if os.path.exists(ipp):
        r2 = json.load(open(ipp)).get("full_r2")
    else:
        r2 = fit.get("r2", fit.get("per_trial_r2"))
    return {
        "kind": kind, "type": "exact", "intro": intro, "tail": tail,
        "top_frags": tfr, "bot_frags": bfr, "shares": [rnd(x, 3) for x in share],
        "L_eff": rnd(neff(g), 1), "dlogit": rnd(dl, 2), "r2": r2f(r2),
        "base_p": rnd(p_base, 3), "p_top": rnd(p_top, 3), "p_bot": rnd(p_bot, 3),
        "dprob": rnd(p_top - p_bot, 3),
        "scatter": {"random": rand, "top": topP, "bot": botP, "tilt": tilt,
                    "rc": rc, "tc": topC, "bc": botC, "zc": tiltC},
    }


def reason_cell(tag, setting, nudge, eff, markers_dir):
    """Reasoning validated cell from reval100.json (+ the selection fit for prompts/pie)."""
    cp = f"data/cells/{tag}/{markers_dir}_{nudge}_{eff}_{setting}"
    rf = f"{cp}/reval100.json"
    if not os.path.exists(rf):
        return None
    d = json.load(open(rf))
    if d.get("degenerate") or "top" not in d or "bot" not in d:
        return None
    p_top = d["top"].get("best_p"); p_bot = d["bot"].get("best_p")
    tc = d["top"].get("choices"); bc = d["bot"].get("choices")
    if p_top is None or p_bot is None or tc is None or bc is None:
        return None
    q = EFFECTS[eff]["question"]
    if nudge == "animals_consider":
        items = load_pool("animals_consider")["items"]
        intro, tail, kind = "Consider these animals:", q, "list"
        tfr = [items[i] for i in tc]; bfr = [items[i] for i in bc]
    else:
        bankf, O = RBANK[nudge]
        groups = [gg[:O] for gg in json.load(open(bankf))["groups"]]
        intro, tail, kind = "", q, "bank"
        tfr = [groups[s][c] for s, c in enumerate(tc)]; bfr = [groups[s][c] for s, c in enumerate(bc)]
    return {
        "kind": kind, "type": "reason", "setting": setting, "dlogit": None,
        "intro": intro, "tail": tail, "top_frags": tfr, "bot_frags": bfr,
        "shares": None, "L_eff": None,
        "base_p": rnd(d.get("base"), 3), "p_top": rnd(p_top, 3), "p_bot": rnd(p_bot, 3),
        "dprob": rnd(p_top - p_bot, 3),
        "n_val": d.get("n_val"), "n_est": d["top"].get("n_est", d.get("n_est")),
        "k_cand": d.get("k_cand"), "k_scr": d.get("k_scr"),
    }


def main():
    cells = {}; models = {}
    # ---- Tier A: exact-logit steering ----
    for tag in sorted(os.listdir("data/cells")):
        if tag not in LB:
            continue
        got = False
        for nu, _, _ in NUDGES:
            for ef, _ in EFFS:
                c = exact_cell(tag, nu, ef)
                if c is not None:
                    cells[f"{tag}|{nu}|{ef}"] = c; got = True
        if got:
            models.setdefault(tag, {"tag": tag, "label": LB[tag], "type": "exact"})
    # ---- Tier B: reasoning validated ----
    REASON = [("qwen3_8b", "B256", "think 256", "thinkcollect"),
              ("qwen3_8b", "B1024", "think 1024", "thinkcollect"),
              ("qwen3_8b", "B4096", "think 4096", "thinkcollect"),
              ("gptoss_20b", "low", "low effort", "effcollect")]
    RNUDGES = [("animals_consider", "animals"), ("phrasing20", "phrasing"),
               ("jsonblob12", "JSON"), ("typos", "typos")]
    for tag, setting, slabel, mk in REASON:
        rtag = f"{tag}::{setting}"; got = False
        for nu, _ in RNUDGES:
            for ef, _ in EFFS:
                c = reason_cell(tag, setting, nu, ef, mk)
                if c is not None:
                    # normalize reasoning nudge key to the canonical family key
                    fam = {"animals_consider": "animals_consider", "phrasing20": "phrasing_L20_O10",
                           "jsonblob12": "jsonblob", "typos": "typos"}[nu]
                    cells[f"{rtag}|{fam}|{ef}"] = c; got = True
        if got:
            base = LB.get(tag, tag)
            models[rtag] = {"tag": rtag, "label": f"{base} · {slabel}", "type": "reason"}

    order = [t for t in ORDER if t in models] + \
            [m for m in models if m not in ORDER and models[m]["type"] == "exact"] + \
            [m for m in models if models[m]["type"] == "reason"]
    out = {
        "nudges": [{"key": n, "label": l, "color": c} for n, l, c in NUDGES],
        "effs": [{"key": e, "label": l} for e, l in EFFS],
        "models": [models[t] for t in order],
        "cells": cells,
        # shared vocab so the client can rebuild any point's prompt from its choices
        "pool": load_pool("animals_consider")["items"],
        "banks": {nu: [gg[:O] for gg in json.load(open(bf))["groups"]] for nu, (bf, O) in BANKS.items()},
    }
    os.makedirs("data", exist_ok=True)
    json.dump(out, open("data/explorer.json", "w"), separators=(",", ":"))
    ncex = sum(1 for c in cells.values() if c["type"] == "exact")
    ncre = sum(1 for c in cells.values() if c["type"] == "reason")
    sz = os.path.getsize("data/explorer.json") / 1e6
    print(f"wrote data/explorer.json  {len(out['models'])} models, "
          f"{ncex} exact cells + {ncre} reason cells  ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
