"""API-key loading for the API-model steering experiments (Section 3.2).

Keys are read from environment variables (recommended -- see .env.example).
As a convenience for local runs, a ``~/.<provider>_key`` file is used as a
fallback. Keys are never committed.
"""
import os
from pathlib import Path

_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "hf": "HF_TOKEN",
}
_FALLBACK = {
    "openai": "~/.openai_key",
    "anthropic": "~/.anthropic_key",
    "gemini": "~/.gemini_key",
    "hf": "~/.hf_token",
}


def get_key(provider: str) -> str:
    env = _ENV[provider]
    val = os.environ.get(env)
    if val:
        return val.strip()
    path = Path(_FALLBACK[provider]).expanduser()
    if path.exists():
        return path.read_text().strip()
    raise RuntimeError(
        f"No {provider} key found. Set ${env} (see .env.example) "
        f"or place it in {_FALLBACK[provider]}."
    )
