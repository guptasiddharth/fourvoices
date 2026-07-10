"""Pluggable Gemma client — stub offline, OpenAI-compatible (Fireworks / vLLM)
on plug-in. Uses the stdlib only (no `openai` package)."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import ssl
import urllib.request

from .config import SETTINGS, Settings
from .styles import Style, generation_prompt


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL_CTX = _ssl_context()


_SYSTEM = ("You output ONLY a single valid JSON object exactly as instructed — no "
           "prose, no markdown fences, no reasoning outside the JSON.")


def _clean(text: str) -> str:
    """Fallback cleaner: strip <think>…</think>, tags, quotes; never empty."""
    if not text:
        return ""
    raw = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    m = re.search(r"<c>(.*?)</c>", text, re.S | re.I)
    if m and m.group(1).strip():
        text = m.group(1)
    return (text.strip().strip('"').strip()) or raw


def _json_field(raw: str, field: str) -> str:
    """Extract one field from a JSON reply; fall back to cleaned text if needed.
    This is what tames reasoning models — the answer lives in the field, the
    brainstorming (if any) stays outside and is discarded."""
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            v = json.loads(m.group()).get(field)
            if v not in (None, ""):
                return str(v).strip().strip('"').strip()
        except Exception:
            pass
    return _clean(raw)


# Offline stand-in for Gemma's video-frame understanding.
_STUB_FACTS = "a person waves at a small dog running across a sunlit park"


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or SETTINGS

    def _chat(self, messages: list[dict], max_tokens: int = 640, json_mode: bool = False) -> str:
        payload = {"model": self.s.model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": 0.4}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}  # structured → no rambling
        req = urllib.request.Request(
            f"{self.s.base_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.s.api_key}",
                     "User-Agent": "FourVoices/0.1"})  # default urllib UA is WAF-blocked
        with urllib.request.urlopen(req, timeout=self.s.request_timeout, context=_SSL_CTX) as resp:
            msg = json.loads(resp.read())["choices"][0]["message"]
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()

    def describe_frames(self, frame_paths: list[str]) -> str:
        """Gemma multimodal → neutral, grounded visual facts for the clip."""
        if self.s.mode == "stub" or not frame_paths:
            return _STUB_FACTS
        content = [{"type": "text", "text":
                    "These frames are sampled in chronological order and span the "
                    "ENTIRE clip from start to finish. Summarize what happens across "
                    "the whole clip: the setting, the main subjects, and how the action "
                    "progresses from beginning to end. 2-4 sentences. Describe only what "
                    "is visibly shown — no invented detail, no dialogue. Output just the "
                    "summary."}]
        for p in frame_paths:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = mimetypes.guess_type(p)[0] or "image/jpeg"
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
        # Plain (no JSON) — vision description is low-creativity and stays clean;
        # avoids any vision+response_format incompatibility on the target model.
        raw = self._chat([{"role": "user", "content": content}], max_tokens=400)
        return _clean(raw)

    def style_caption(self, facts: str, style: Style) -> str:
        if self.s.mode == "stub":
            return style.stub_template.format(facts=facts.rstrip("."))
        raw = self._chat([{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": generation_prompt(facts, style)}],
                         json_mode=True)
        return _json_field(raw, "caption")

    def check_tone(self, caption: str, style: Style) -> bool:
        """LLM tone verification (mirrors the judge). Stub trusts the template."""
        if self.s.mode == "stub":
            return True
        raw = self._chat([{"role": "user", "content":
            f'Return JSON {{"match": true|false}}: is this caption clearly in a '
            f'{style.name} tone ({style.guidance})?\n\nCaption: {caption}'}],
            max_tokens=300, json_mode=True)
        mo = re.search(r"\{.*\}", raw, re.S)
        try:
            return bool(json.loads(mo.group()).get("match")) if mo else True
        except Exception:
            return True   # don't force needless regeneration on a parse hiccup
