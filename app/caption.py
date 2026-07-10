"""Pipeline: clip → frames → grounded facts → four styled captions → self-check.

Ground once, style four times. The self-check enforces the two things the
hackathon's LLM-judge scores — accuracy (caption stays faithful to the grounded
facts) and tone (each caption matches its target voice) — and regenerates any
caption that drifts.
"""

from __future__ import annotations

from typing import Any

from .frames import sample_frames
from .llm import LLMClient
from .styles import STYLES, accuracy_ok, distinct_ok


class VideoCaptioner:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def caption(self, clip_path: str | None = None, *, facts: str | None = None) -> dict[str, Any]:
        """Caption one clip. Pass `clip_path` (real video) or `facts` (offline/testing)."""
        frames = sample_frames(clip_path, self.llm.s.n_frames) if clip_path else []
        grounded = facts or self.llm.describe_frames(frames)

        captions: dict[str, str] = {}
        checks: dict[str, dict[str, bool]] = {}
        for style in STYLES:
            text = self.llm.style_caption(grounded, style)
            acc = accuracy_ok(grounded, text)
            tone = self.llm.check_tone(text, style)
            if not (acc and tone):                      # one regeneration attempt
                text = self.llm.style_caption(grounded, style)
                acc = accuracy_ok(grounded, text)
                tone = self.llm.check_tone(text, style)
            captions[style.key] = text
            checks[style.key] = {"accuracy": acc, "tone": tone}

        return {
            "clip": clip_path,
            "grounded_facts": grounded,
            "captions": captions,
            "checks": checks,
            "distinct": distinct_ok(captions),
            "n_frames": len(frames),
        }
