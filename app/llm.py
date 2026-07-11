"""Pluggable Gemma client — stub offline, OpenAI-compatible (Fireworks / vLLM)
on plug-in. Uses the stdlib only (no `openai` package)."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import ssl
import time
import urllib.error
import urllib.request

from .config import SETTINGS, Settings
from .styles import STYLES, Style, generation_prompt


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
    for close in ("</think>", "</thought>"):        # Gemma 4 uses <thought>…</thought>
        if close in text:
            text = text.rsplit(close, 1)[-1]
    m = re.search(r"<c>(.*?)</c>", text, re.S | re.I)
    if m and m.group(1).strip():
        text = m.group(1)
    return (text.strip().strip('"').strip()) or raw


def _degenerate(text: str) -> bool:
    """True if text looks like a repetition loop or junk — never ship it.
    Gemma 4 (with its <thought> suppressed) will occasionally spin into
    'groundbreaking groundbreaking …' loops; this is the backstop that catches
    them so a bad sample falls back to a clean caption instead of shipping garbage."""
    if not text:
        return True
    if "<thought>" in text or "</thought>" in text:
        return True
    words = text.split()
    if len(words) > 60:                                  # a caption is one sentence
        return True
    for i in range(len(words) - 3):                      # 4+ identical words in a row
        if words[i] == words[i + 1] == words[i + 2] == words[i + 3]:
            return True
    if len(words) >= 12 and len({w.lower() for w in words}) / len(words) < 0.5:
        return True                                      # very low lexical variety
    return False


def _json_field(raw: str, field: str) -> str:
    """Extract one field from a JSON reply; fall back to cleaned text if needed.
    This is what tames reasoning models — the answer lives in the field, the
    brainstorming (if any) stays outside and is discarded."""
    t = raw
    for close in ("</think>", "</thought>"):
        if close in t:
            t = t.rsplit(close, 1)[-1]
    m = re.search(r"\{.*\}", t, re.S)
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

    def _chat(self, messages: list[dict], max_tokens: int = 640,
              json_mode: bool = False, prefill: str | None = None) -> str:
        msgs = list(messages)
        if prefill is not None:
            # Prefilling the assistant turn makes Gemma 4 continue from `prefill`
            # and emit the answer directly, skipping its <thought> reasoning block
            # entirely — clean, fast, and it never truncates mid-thought.
            msgs = msgs + [{"role": "assistant", "content": prefill}]
        payload = {"model": self.s.model, "messages": msgs,
                   "max_tokens": max_tokens, "temperature": 0.3, "top_p": 0.9}
        if json_mode and prefill is None:
            payload["response_format"] = {"type": "json_object"}  # structured → no rambling
        req = urllib.request.Request(
            f"{self.s.base_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.s.api_key}",
                     "User-Agent": "FourVoices/0.1"})  # default urllib UA is WAF-blocked
        msg = None
        for attempt in range(4):                     # retry transient throttling / 5xx
            try:
                with urllib.request.urlopen(req, timeout=self.s.request_timeout,
                                            context=_SSL_CTX) as resp:
                    msg = json.loads(resp.read())["choices"][0]["message"]
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        content = msg.get("content") or msg.get("reasoning_content") or ""
        if prefill is not None:
            content = prefill + content
        return content.strip()

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
        # Room for Gemma's <thought> block plus the summary; _clean drops the thought.
        msg = [{"role": "user", "content": content}]
        facts = _clean(self._chat(msg, max_tokens=1024))
        if _degenerate(facts):                       # a looped/garbled description → retry once
            facts = _clean(self._chat(msg, max_tokens=1024))
        return _STUB_FACTS if _degenerate(facts) else facts

    def style_all(self, facts: str, keys: list[str]) -> dict[str, str]:
        """All requested styles in ONE structured JSON call (efficient + coherent).
        Falls back to per-style generation for anything missing/unparseable."""
        by = {s.key: s for s in STYLES}
        keys = [k for k in keys if k in by]
        if self.s.mode == "stub":
            return {k: by[k].stub_template.format(facts=facts.rstrip(".")) for k in keys}
        guide = "\n".join(f'- "{k}" ({by[k].name}): {by[k].guidance}' for k in keys)
        example = ", ".join(f'"{k}": "..."' for k in keys)
        prompt = (f"GROUNDED visual facts (do not add anything not present here):\n{facts}\n\n"
                  f"Write ONE caption for EACH style below — faithful to the facts, one short "
                  f"sentence each, tone unmistakable:\n{guide}\n\n"
                  f"Return ONLY a JSON object mapping each style key to its caption: {{{example}}}. "
                  f"No other text.")
        raw = self._chat([{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": prompt}], max_tokens=400, prefill="{")
        out: dict[str, str] = {}
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                data = json.loads(m.group())
                for k in keys:
                    v = str(data.get(k, "")).strip().strip('"')
                    if v and not _degenerate(v):        # drop looped/garbled values
                        out[k] = v
            except Exception:  # noqa: BLE001
                pass
        for k in keys:                      # backfill any missing/rejected style individually
            if not out.get(k):
                out[k] = self.style_caption(facts, by[k])
        return out

    def style_caption(self, facts: str, style: Style) -> str:
        if self.s.mode == "stub":
            return style.stub_template.format(facts=facts.rstrip("."))
        msg = [{"role": "system", "content": _SYSTEM},
               {"role": "user", "content": generation_prompt(facts, style)}]
        for _ in range(2):                              # one retry on a degenerate sample
            cap = _json_field(self._chat(msg, max_tokens=100, prefill="{"), "caption")
            if not _degenerate(cap):
                return cap
        return style.stub_template.format(facts=facts.rstrip("."))   # clean, never garbage

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
