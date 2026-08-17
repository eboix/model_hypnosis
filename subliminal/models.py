"""Model registry: tag -> (HF id, tensor_parallel_size).

Tags are the canonical short names used in data/cells/<tag>/ and figures.
"""

MODELS = {
    # Qwen2.5 size ladder (Apache-2.0, ungated)
    "qwen25_05b": ("Qwen/Qwen2.5-0.5B-Instruct", 1),
    "qwen25_15b": ("Qwen/Qwen2.5-1.5B-Instruct", 1),
    "qwen25_3b":  ("Qwen/Qwen2.5-3B-Instruct", 1),
    "qwen25_7b":  ("Qwen/Qwen2.5-7B-Instruct", 1),
    "qwen25_14b": ("Qwen/Qwen2.5-14B-Instruct", 1),
    "qwen25_32b": ("Qwen/Qwen2.5-32B-Instruct", 2),
    "qwen25_72b": ("Qwen/Qwen2.5-72B-Instruct", 2),
    # cross-family ~7-14B
    "llama31_8b": ("meta-llama/Llama-3.1-8B-Instruct", 1),   # gated (granted)
    "gemma2_9b":  ("google/gemma-2-9b-it", 1),               # gated (granted)
    "phi4":       ("microsoft/phi-4", 1),                    # MIT
    "mistral7b":  ("mistralai/Mistral-7B-Instruct-v0.3", 1), # gated (check!)
    "olmo2_7b":   ("allenai/OLMo-2-1124-7B-Instruct", 1),    # Apache, ungated
    # Qwen3 family, non-thinking use (hybrids via enable_thinking=False;
    # -2507 Instruct variants are natively non-thinking, no template kwarg)
    "qwen3_4b_i2507": ("Qwen/Qwen3-4B-Instruct-2507", 1),
    "qwen3_14b":      ("Qwen/Qwen3-14B", 1),
    "qwen3_30b_i2507": ("Qwen/Qwen3-30B-A3B-Instruct-2507", 1),  # MoE, needs H200
    # --- require VLLM_ENV=vllm-new (vllm 0.11.2): OLMo 3 ---
    # --- require the vllm026 container (run_container.sh): Gemma 4, Qwen3.5, gpt-oss ---
    "gemma4_12b":  ("google/gemma-4-12B-it", 1),
    "gemma4_e4b":  ("google/gemma-4-E4B-it", 1),
    "gemma4_e2b":  ("google/gemma-4-E2B-it", 1),
    "gemma4_31b":  ("google/gemma-4-31B-it", 2),
    "olmo3_7b":    ("allenai/Olmo-3-7B-Instruct", 1),
    "olmo3_7b_think": ("allenai/Olmo-3-7B-Think", 1),
    # NOTE: no Olmo-3-32B-Instruct exists on HF (verified 2026-08-03) — 32B is Think-only
    "olmo3_32b_think": ("allenai/Olmo-3-32B-Think", 2),
    "qwen35_4b":   ("Qwen/Qwen3.5-4B", 1),
    "qwen35_9b":   ("Qwen/Qwen3.5-9B", 1),
    "qwen35_27b":  ("Qwen/Qwen3.5-27B", 2),
    # reasoning-capable (sampled-CoT protocol; logistic fit on binary outcomes)
    # Qwen3 hybrid ladder (thinking on/off per prompt; budget caps via
    # sample_binary_budget forced </think> exit)
    "qwen3_06b":  ("Qwen/Qwen3-0.6B", 1),
    "qwen3_17b":  ("Qwen/Qwen3-1.7B", 1),
    "qwen3_4b":   ("Qwen/Qwen3-4B", 1),
    "qwen3_8b":   ("Qwen/Qwen3-8B", 1),                      # thinking on/off pair
    "qwen3_32b":  ("Qwen/Qwen3-32B", 2),
    "gptoss_20b": ("openai/gpt-oss-20b", 1),                 # VLLM_ENV=vllm-new; effort low/med/high via chat_template_kwargs
    "r1_llama8b": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 1),  # pairs with llama31_8b
    "r1_qwen15b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1), # pairs with qwen25_15b
    "r1_qwen7b":  ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 1),   # pairs with qwen25_7b
    "r1_qwen14b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 1),  # pairs with qwen25_14b
    "r1_qwen32b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 2),  # pairs with qwen25_32b
    "gptoss_120b": ("openai/gpt-oss-120b", 1),               # container + H200 only
}


def resolve(tag):
    if tag not in MODELS:
        raise KeyError(f"unknown model tag {tag!r}; known: {sorted(MODELS)}")
    return MODELS[tag]
