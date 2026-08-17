"""vLLM backend: exact two-token logit reads for forced binary choices.

One LocalModel instance per (model, GPU allocation); the answer-token pair is
per-call, so a single loaded model serves every effect (5v7, trolley, ...).

Reads are exact because the forced-choice prompt concentrates the next-token
distribution on the two answer tokens, which therefore always sit inside the
top-k logprobs (k=20). Deterministic under fixed seed; temperature=1.0 so the
reported logprobs are the true softmax.
"""
import math
import os

from vllm import LLM, SamplingParams


def _tokprompt(ids):
    # Coerce every id to a plain Python int: the container models' newer
    # tokenizers return numpy ints, which vLLM 0.26's strict token-prompt
    # validation rejects ("should be a list of integers") and whose presence
    # also breaks its max()-over-prompt-ids check ("int > str").
    return {"prompt_token_ids": [int(x) for x in ids]}


class LocalModel:
    def __init__(self, model_id, tp=1, dtype="bfloat16", max_model_len=1024,
                 gpu_mem_util=0.90, seed=0, enforce_eager=True,
                 chat_kwargs=None):
        self.model_id = model_id
        self.chat_kwargs = chat_kwargs or {}
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tp,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            seed=seed,
            # No download_dir: with it set, vLLM ignores the standard HF cache
            # ($HF_HOME/hub) and re-downloads models to that dir -> every model
            # was fetched twice. Leaving it unset makes vLLM reuse the
            # predownloaded snapshot_download cache.
            trust_remote_code=True,
            enforce_eager=enforce_eager,
        )
        # max_tokens=2: SentencePiece models (Mistral) sometimes emit a bare
        # whitespace piece first and the answer digit second; read_batch falls
        # back to position 2 in that case. Other models match at position 1.
        self.sp = SamplingParams(max_tokens=2, temperature=1.0, logprobs=20, seed=seed)
        self.tok = self.llm.get_tokenizer()
        self.n_missing = 0  # trials where neither answer token appeared in top-k

    def assert_single_token(self, *words):
        """An answer word is usable if SOME single vocab piece decodes to it —
        checked via common piece spellings (bare, SentencePiece '▁', BPE 'Ġ').
        encode() length is the wrong test: Mistral's SP vocab has a bare '5'
        piece the model can emit, but encode('5') still yields two tokens."""
        unk = getattr(self.tok, "unk_token_id", None)
        for w in words:
            ok = False
            for cand in (w, "▁" + w, "Ġ" + w):
                tid = self.tok.convert_tokens_to_ids(cand)
                if tid is not None and tid >= 0 and tid != unk:
                    ok = True
                    break
            ok = ok or any(len(self.tok.encode(v, add_special_tokens=False)) == 1
                           for v in (w, " " + w))
            if not ok:
                raise ValueError(
                    f"no single vocab piece decodes to {w!r} for {self.model_id}")

    def assert_prefix_readable(self, tok_a, tok_b):
        """Multi-token answer words are readable at position 1 iff their first
        BPE pieces are distinct and non-overlapping as string prefixes."""
        pa = self.tok.decode([self.tok.encode(" " + tok_a,
                                              add_special_tokens=False)[0]]).strip().lower()
        pb = self.tok.decode([self.tok.encode(" " + tok_b,
                                              add_special_tokens=False)[0]]).strip().lower()
        if not pa or not pb or pa.startswith(pb) or pb.startswith(pa):
            raise ValueError(
                f"first-piece prefixes collide for {tok_a!r}/{tok_b!r} "
                f"({pa!r} vs {pb!r}) on {self.model_id}")

    def read_batch(self, user_texts, tok_a, tok_b, prefix=False):
        """P(tok_a)/(P(tok_a)+P(tok_b)) of the first generated token, per text.

        Matching is casefolded and whitespace-stripped, and sums over every
        top-k entry that decodes to the answer word (e.g. 'red', ' Red', 'RED').
        With prefix=True (multi-token answers like 'aquamarine'), a top-k entry
        also matches if it is a >=2-char prefix of exactly one answer word.
        Returns None for a text where neither token is in the top-k.
        """
        a, b = tok_a.strip().lower(), tok_b.strip().lower()
        convs = [[{"role": "user", "content": t}] for t in user_texts]
        kw = {"chat_template_kwargs": self.chat_kwargs} if self.chat_kwargs else {}
        outs = self.llm.chat(convs, self.sp, use_tqdm=False, **kw)
        res = []
        for o in outs:
            comp = o.outputs[0]
            pl = self._read_pos(comp.logprobs[0], a, b, prefix)
            if pl is None and len(comp.logprobs) > 1:
                # SP models may emit a pure-whitespace piece first; the answer
                # distribution then lives at position 2 (conditioned on it).
                first = self.tok.decode([comp.token_ids[0]])
                if first.strip() == "":
                    pl = self._read_pos(comp.logprobs[1], a, b, prefix)
            if pl is None:
                self.n_missing += 1
            res.append(pl)
        return res

    def read_batch_forced(self, user_texts, tok_a, tok_b):
        """Exact P(tok_a)/(P(tok_a)+P(tok_b)) at the first generated position,
        REGARDLESS of rank. read_batch only sees the top-20 logprobs, so it
        returns None whenever the model would rather open with "As an AI...";
        here each answer token is scored directly by appending it to the
        prompt and reading its prompt-logprob, which is exact even when the
        token sits far down the distribution.

        This measures the model's internal lean conditional on answering with
        one of the two tokens -- the same quantity read_batch reports -- and
        works for prompts that get no format instruction. It does NOT imply the
        model would actually emit either token unprompted.
        """
        # every vocab spelling of each answer word (bare, SentencePiece '▁',
        # BPE 'Ġ'); their probabilities are SUMMED, because e.g. Mistral emits
        # '▁5' where Qwen emits '5' -- scoring only one spelling reads the
        # wrong path and can be off by many logits.
        unk = getattr(self.tok, "unk_token_id", None)
        ids = {}
        for w in (tok_a, tok_b):
            cands = []
            # CASE MATTERS: models emit "No"/"Yes" at a sentence start, so
            # scoring only the lowercase piece reads a rare off-path token and
            # can be many logits wrong. read_batch casefolds its top-k matches;
            # mirror that here by scoring every case/prefix spelling.
            forms = {w, w.lower(), w.capitalize(), w.upper()}
            for cand in [f(x) for x in forms for f in
                         (lambda z: z, lambda z: "▁" + z, lambda z: "Ġ" + z)]:
                t = self.tok.convert_tokens_to_ids(cand)
                if t is not None and t >= 0 and t != unk and t not in cands:
                    cands.append(t)
            if not cands:
                enc = self.tok.encode(w, add_special_tokens=False)
                if len(enc) != 1:
                    raise ValueError(f"{w!r} is not a single token for {self.model_id}")
                cands = [enc[0]]
            ids[w] = cands

        prefixes = []
        for t in user_texts:
            kw = dict(self.chat_kwargs) if self.chat_kwargs else {}
            enc = self.tok.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=True,
                add_generation_prompt=True, **kw)
            # Newer transformers (container) returns a BatchEncoding/dict instead
            # of a flat id list; unwrap to the token ids and drop any batch nesting.
            if isinstance(enc, dict) or hasattr(enc, "input_ids"):
                enc = enc["input_ids"]
            if len(enc) and isinstance(enc[0], (list, tuple)):
                enc = enc[0]
            prefixes.append(list(enc))

        # SentencePiece models (Mistral) usually emit a bare whitespace piece
        # first and the digit second. Scoring only position 1 would read an
        # off-path token, so when such a piece exists we also marginalise over
        # that path:  P(w) = P(w at pos1) + P(ws at pos1) * P(w at pos2 | ws).
        ws_id = self.tok.convert_tokens_to_ids("▁")
        if ws_id is None or ws_id < 0 or ws_id == unk:
            ws_id = None

        sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
        flat, spans = [], []
        for pre in prefixes:
            start = len(flat)
            for w in (tok_a, tok_b):
                for t in ids[w]:
                    flat.append(_tokprompt(list(pre) + [t]))
            if ws_id is not None:
                for w in (tok_a, tok_b):
                    flat.append(_tokprompt(list(pre) + [ws_id, ids[w][0]]))
            spans.append((start, len(ids[tok_a]), len(ids[tok_b])))
        outs = self.llm.generate(flat, sp, use_tqdm=False)

        res = []
        for start, na, nb in spans:
            pa = sum(math.exp(outs[start + j].prompt_logprobs[-1]
                              [ids[tok_a][j]].logprob) for j in range(na))
            pb = sum(math.exp(outs[start + na + j].prompt_logprobs[-1]
                              [ids[tok_b][j]].logprob) for j in range(nb))
            if ws_id is not None:
                k = start + na + nb
                for j, (w, acc) in enumerate(((tok_a, "a"), (tok_b, "b"))):
                    o = outs[k + j]
                    p_ws = math.exp(o.prompt_logprobs[-2][ws_id].logprob)
                    p_dig = math.exp(o.prompt_logprobs[-1][ids[w][0]].logprob)
                    if acc == "a":
                        pa += p_ws * p_dig
                    else:
                        pb += p_ws * p_dig
            l = math.log(pa) - math.log(pb)
            res.append((1.0 / (1.0 + math.exp(-l)), l))
        return res

    def sample_binary_batch(self, user_texts, tok_a, tok_b, max_tokens=1200,
                            temperature=0.6, top_p=0.95, seed0=0,
                            thinking=None, open_marker="<think>",
                            close_marker="</think>", skip_special=True):
        """Reasoning-model protocol: sample a full (CoT +) answer per prompt and
        parse the final answer to y in {1 (tok_a), 0 (tok_b), None}. Per-request
        seeds keep runs deterministic yet varied across trials. `thinking`
        toggles chat-template thinking where supported (Qwen3/Qwen3.5/Gemma4).
        Markers default to Qwen <think> tags; Gemma 4 uses
        open="<|channel>thought", close="<channel|>" with skip_special=False
        (its markers are special tokens and would otherwise be stripped)."""
        a, b = tok_a.strip().lower(), tok_b.strip().lower()
        convs = [[{"role": "user", "content": t}] for t in user_texts]
        sps = [SamplingParams(max_tokens=max_tokens, temperature=temperature,
                              top_p=top_p, seed=seed0 + i,
                              skip_special_tokens=skip_special)
               for i in range(len(convs))]
        kw = {}
        if thinking is not None:
            kw["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        outs = self.llm.chat(convs, sps, use_tqdm=False, **kw)
        ys = []
        for o in outs:
            text = o.outputs[0].text
            if close_marker in text:
                text = text.rsplit(close_marker, 1)[1]
            elif open_marker in text:
                ys.append(None)  # truncated mid-thought
                continue
            ys.append(self._parse_last(text, a, b))
        return ys

    @staticmethod
    def _parse_last(text, a, b):
        """Last standalone occurrence wins: CoT answers often restate both
        options ("between 5 and 7, I pick 5") -- the final mention is the
        answer. Rejecting both-present texts nonrandomly dropped trials."""
        words = [w.strip(".,!?:;()*'\"").lower() for w in text.split()]
        for w in reversed(words):
            if w == a:
                return 1
            if w == b:
                return 0
        return None

    # Qwen3 report's recommended early-exit injection for budgeted thinking.
    QWEN_THINK_EXIT = ("\n\nConsidering the limited time by the user, I have "
                      "to give the solution based on the thinking directly "
                      "now.\n</think>\n\n")

    # marker sets for budget-capped thinking, per model family
    THINK_MARKERS = {
        "qwen": {"close": "</think>",
                 "exit": ("\n\nConsidering the limited time by the user, I have "
                          "to give the solution based on the thinking directly "
                          "now.\n</think>\n\n")},
        # Gemma 4 emits <|channel>thought ... <channel|>answer ; force-close the
        # thought channel to exit early.
        "gemma": {"close": "<channel|>", "exit": "\n<channel|>"},
    }

    def sample_binary_budget(self, user_texts, tok_a, tok_b, budget,
                             answer_tokens=512, temperature=0.6, top_p=0.95,
                             seed0=0, markers="qwen"):
        """Thinking with a hard token budget: phase 1 generates at most `budget`
        tokens with thinking on; any completion still inside the think channel
        gets the family's exit string force-appended and continues for
        `answer_tokens` more, so every trial terminates with a readable answer.
        `markers` selects the close-tag/exit pair (qwen | gemma).

        Returns (ys, phase1_toks, capped)."""
        a, b = tok_a.strip().lower(), tok_b.strip().lower()
        mk = self.THINK_MARKERS[markers]
        close, exit_s = mk["close"], mk["exit"]
        # Text prompts, not prompt_token_ids: vLLM 0.26's token-id validation
        # crashes on some tokenizers (Qwen3.5: max_token_id is a str). vLLM
        # re-adds BOS when tokenizing text, so strip a template-included one.
        bos = getattr(self.tok, "bos_token", None)
        pre_txt = []
        for t in user_texts:
            try:
                s = self.tok.apply_chat_template(
                    [{"role": "user", "content": t}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True)
            except TypeError:               # template lacks enable_thinking kwarg
                s = self.tok.apply_chat_template(
                    [{"role": "user", "content": t}], tokenize=False,
                    add_generation_prompt=True)
            if bos and s.startswith(bos):
                s = s[len(bos):]
            pre_txt.append(s)
        sps = [SamplingParams(max_tokens=budget, temperature=temperature,
                              top_p=top_p, seed=seed0 + i,
                              skip_special_tokens=False)
               for i in range(len(pre_txt))]
        outs = self.llm.generate(pre_txt, sps, use_tqdm=False)
        n = len(user_texts)
        ys, toks, capped = [None] * n, [0] * n, [False] * n
        cont = []  # (index, continuation prompt text, forced?)
        for i, o in enumerate(outs):
            c = o.outputs[0]
            text, toks[i] = c.text, len(c.token_ids)
            if close in text:
                y = self._parse_last(text.rsplit(close, 1)[1], a, b)
                if y is None and c.finish_reason == "length":
                    cont.append((i, pre_txt[i] + text, False))
                else:
                    ys[i] = y
            else:
                capped[i] = True
                cont.append((i, pre_txt[i] + text + exit_s, True))
        if cont:
            sps2 = [SamplingParams(max_tokens=answer_tokens,
                                   temperature=temperature, top_p=top_p,
                                   seed=seed0 + 50_000 + j,
                                   skip_special_tokens=False)
                    for j in range(len(cont))]
            outs2 = self.llm.generate([txt for _, txt, _ in cont],
                                      sps2, use_tqdm=False)
            for (i, _, _), o2 in zip(cont, outs2):
                t2 = o2.outputs[0].text
                if close in t2:
                    t2 = t2.rsplit(close, 1)[1]
                ys[i] = self._parse_last(t2, a, b)
        return ys, toks, capped

    def sample_binary_effort(self, user_texts, tok_a, tok_b, effort,
                             max_tokens=3000, temperature=0.6, top_p=0.95,
                             seed0=0):
        """Harmony (gpt-oss) analog of sample_binary_budget: the thinking amount
        is set by `reasoning_effort` (low|medium|high) via the chat template, not
        a token cap. CoT lives in the analysis channel, the answer in the final
        channel (<|channel|>final<|message|>). Parses the last answer token in
        the final channel. Returns (ys, cats, n_toks)."""
        a, b = tok_a.strip().lower(), tok_b.strip().lower()
        FINAL = "<|channel|>final<|message|>"
        convs = [[{"role": "user", "content": t}] for t in user_texts]
        sps = [SamplingParams(max_tokens=max_tokens, temperature=temperature,
                              top_p=top_p, seed=seed0 + i,
                              skip_special_tokens=False)
               for i in range(len(convs))]
        outs = self.llm.chat(convs, sps, use_tqdm=False,
                             chat_template_kwargs={"reasoning_effort": effort})
        ys, cats, ntoks = [], [], []
        for o in outs:
            c = o.outputs[0]
            ntoks.append(len(c.token_ids))
            if FINAL in c.text:
                post = c.text.rsplit(FINAL, 1)[1]
                post = post.replace("<|return|>", " ").replace("<|end|>", " ")
                y = self._parse_last(post, a, b)
                ys.append(y)
                cats.append("parsed" if y is not None else "final_no_token")
            else:
                ys.append(None)
                cats.append("truncated" if c.finish_reason == "length"
                            else "no_final")
        return ys, cats, ntoks

    @staticmethod
    def _read_pos(lp, a, b, prefix=False):
        """(p, l) with l = exact log-odds log P(a) - log P(b). If one token
        fell out of the top-k, l is a CENSORED bound using the k-th logprob
        as the missing side's ceiling (better than a silent p=1.0)."""
        def match(t, w):
            return t == w or (prefix and len(t) >= 2 and w.startswith(t))
        pa = pb = 0.0
        for info in lp.values():
            t = (info.decoded_token or "").strip().lower()
            if match(t, a):
                pa += math.exp(info.logprob)
            elif match(t, b):
                pb += math.exp(info.logprob)
        if pa == 0.0 and pb == 0.0:
            return None
        if pa > 0.0 and pb > 0.0:
            l = math.log(pa) - math.log(pb)
        else:
            floor = min(info.logprob for info in lp.values())
            l = (math.log(pa) - floor) if pa > 0.0 else (floor - math.log(pb))
        return 1.0 / (1.0 + math.exp(-l)), l
