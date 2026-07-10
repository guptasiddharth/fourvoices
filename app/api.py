"""FastAPI service.

    POST /caption      {clip_path?} or {facts?}   → 4 styled captions + checks
    GET  /health

Run: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from .caption import VideoCaptioner
from .config import SETTINGS

app = FastAPI(title="FourVoices Video Captioner", version="0.1.0",
              description="Grounded four-style video captioning with Gemma.")
_engine = VideoCaptioner()


class CaptionRequest(BaseModel):
    clip_path: str | None = None
    facts: str | None = None      # supply directly to caption without a video (testing)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "llm_mode": SETTINGS.mode, "model": SETTINGS.model,
            "endpoint": SETTINGS.base_url or "(stub — no endpoint configured)"}


@app.post("/caption")
def caption(req: CaptionRequest) -> dict[str, Any]:
    return _engine.caption(clip_path=req.clip_path, facts=req.facts)
