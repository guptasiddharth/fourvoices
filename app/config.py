"""Settings resolved from env. Runs in offline `stub` mode with no keys; goes
live the moment `LLM_BASE_URL` points at Fireworks (Gemma) or a vLLM Gemma
server on an AMD MI300X."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    mode: str                 # "auto" | "stub" | "openai"
    base_url: str | None
    api_key: str
    model: str                # a Gemma multimodal model id
    n_frames: int             # frames sampled per clip
    request_timeout: float


def load_settings() -> Settings:
    base_url = os.getenv("LLM_BASE_URL") or None
    api_key = os.getenv("LLM_API_KEY", "EMPTY")
    mode = os.getenv("VC_LLM_MODE", "auto").lower()
    if mode == "auto":
        # Go live only when we have BOTH an endpoint and a real key; otherwise fall
        # back to offline stub captions (valid output) instead of erroring every call.
        mode = "openai" if (base_url and api_key not in ("", "EMPTY")) else "stub"
    return Settings(
        mode=mode,
        base_url=base_url,
        api_key=api_key,
        # Gemma 4 (Apr 2026) is natively image+video multimodal. Validated live on
        # Google's free OpenAI-compatible endpoint as `gemma-4-26b-a4b-it` (26B MoE,
        # vision). Override via LLM_MODEL for any other Gemma host.
        model=os.getenv("LLM_MODEL", "gemma-4-26b-a4b-it"),
        n_frames=int(os.getenv("VC_N_FRAMES", "12")),  # floor; scales up with clip length
        request_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
    )


SETTINGS = load_settings()
