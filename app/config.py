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
    mode = os.getenv("VC_LLM_MODE", "auto").lower()
    if mode == "auto":
        mode = "openai" if base_url else "stub"
    return Settings(
        mode=mode,
        base_url=base_url,
        api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        # Gemma 4 (Apr 2026) is natively video+image multimodal — ideal for this
        # track. Confirm the exact Fireworks id at launch; gemma-3-27b-it is the
        # known-working multimodal fallback.
        model=os.getenv("LLM_MODEL", "accounts/fireworks/models/gemma-4-26b-a4b-it"),
        n_frames=int(os.getenv("VC_N_FRAMES", "8")),   # floor; scales up with clip length
        request_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
    )


SETTINGS = load_settings()
