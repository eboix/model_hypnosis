import sys, os, json
from subliminal.backend import LocalModel
from subliminal.effects import EFFECTS
from subliminal.models import resolve
d = json.load(open("data/intro_phrasing_mini_qwen25_14b_trolley.json"))
q = EFFECTS["trolley"]["question"]
prompts = [d["top"]["para"] + " " + q, d["bot"]["para"] + " " + q]
tags = os.environ["TAGS"].split(",")
for tag in tags:
    mid, tp = resolve(tag)
    try:
        m = LocalModel(mid, tp=tp, max_model_len=2048)
        (pt, lt), (pb, lb) = m.read_batch(prompts, "2", "1")
        print(f"[pairtransfer] {tag:<12} A(P_agree)={pt:.4f}  B={pb:.4f}  gap={lt-lb:+.2f} logits",
              flush=True)
        del m
        import gc, torch; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"[pairtransfer] {tag:<12} FAILED: {str(e)[:80]}", flush=True)
