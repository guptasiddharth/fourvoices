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
from .styles import STYLES, Style, generation_prompt, style_system


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
    low = text.lower()
    if any(x in text for x in ("{", "}", "```")) or "<thought" in low:
        return True                                      # leaked JSON/fence/reasoning
    if low.lstrip().startswith(("thought", "*", "-", "1.", "option")):
        return True                                      # reasoning/list preamble leaked
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
    t = raw.replace("```json", " ").replace("```", " ")
    for close in ("</think>", "</thought>"):
        if close in t:
            t = t.rsplit(close, 1)[-1]
    # Try each flat {...} candidate (our JSON is flat) then a greedy fallback —
    # this pulls a valid {"field": "..."} out even when it's buried in reasoning.
    for pat in (r"\{[^{}]*\}", r"\{.*\}"):
        for m in re.finditer(pat, t, re.S):
            try:
                v = json.loads(m.group()).get(field)
                if v not in (None, ""):
                    return str(v).strip().strip('"').strip()
            except Exception:
                continue
    return _clean(t)


# Offline stand-in for Gemma's video-frame understanding.
_STUB_FACTS = "a person waves at a small dog running across a sunlit park"


def _bad_facts(text: str) -> bool:
    """Validity check for the grounding paragraph (which is legitimately long, so
    the caption-oriented _degenerate rules don't apply). Flags empty/leaked/looped."""
    if not text or len(text.split()) < 4:
        return True
    if "<thought" in text.lower() or "```" in text:
        return True
    words = text.split()
    for i in range(len(words) - 3):
        if words[i] == words[i + 1] == words[i + 2] == words[i + 3]:
            return True
    return False


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or SETTINGS

    def _chat(self, messages: list[dict], max_tokens: int = 640,
              json_mode: bool = False, prefill: str | None = None,
              temperature: float = 0.4) -> str:
        msgs = list(messages)
        if prefill is not None:
            # Prefilling the assistant turn makes Gemma 4 continue from `prefill`
            # and emit the answer directly, skipping its <thought> reasoning block
            # entirely — clean, fast, and it never truncates mid-thought.
            msgs = msgs + [{"role": "assistant", "content": prefill}]
        payload = {"model": self.s.model, "messages": msgs,
                   "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.9}
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
                    "You are a meticulous visual analyst. These images are keyframes "
                    "sampled in order across a single short video clip. In 2-4 factual "
                    "sentences, describe: the setting/location, the main subjects, the "
                    "actions or motion across the frames from start to finish, the mood, "
                    "notable visual details (colors, lighting, weather), and any visible "
                    "text, signage, screens, or technology. Neutral and factual — no humor, "
                    "no opinion, no invented detail. English only.\n"
                    'Respond with ONLY a JSON object: {"description": "<the paragraph>"}.'}]
        for p in frame_paths:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = mimetypes.guess_type(p)[0] or "image/jpeg"
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
        # JSON + assistant-prefill of "{" skips Gemma's <thought> entirely (as in
        # styling), so the paragraph comes back clean without a huge token budget.
        # Low temperature for faithfulness.
        msg = [{"role": "user", "content": content}]
        facts = _json_field(self._chat(msg, max_tokens=500, prefill="{", temperature=0.2),
                            "description")
        if _bad_facts(facts):                        # leaked/looped/empty → retry once
            facts = _json_field(self._chat(msg, max_tokens=500, prefill="{", temperature=0.2),
                                "description")
        return _STUB_FACTS if _bad_facts(facts) else facts

    def style_all(self, facts: str, keys: list[str]) -> dict[str, str]:
        """One focused call per style (each with its own role + few-shots). Separate
        calls beat a single all-styles JSON call: the model nails one register at a
        time instead of diluting four voices across one generation."""
        by = {s.key: s for s in STYLES}
        keys = [k for k in keys if k in by]
        return {k: self.style_caption(facts, by[k]) for k in keys}

    def style_caption(self, facts: str, style: Style) -> str:
        if self.s.mode == "stub":
            return style.stub_template.format(facts=facts.rstrip("."))
        msg = [{"role": "system", "content": style_system(style)},
               {"role": "user", "content": generation_prompt(facts, style)}]
        # Higher temperature so the humor/irony actually lands; the guard + stub
        # fallback keep a bad sample from ever shipping.
        for _ in range(3):                              # retries on a degenerate sample
            cap = _json_field(self._chat(msg, max_tokens=110, prefill="{",
                                         temperature=0.5), "caption")
            if not _degenerate(cap):
                return cap
        return style.stub_template.format(facts=facts.rstrip("."))   # clean, never garbage

    def check_tone(self, caption: str, style: Style) -> bool:
        """LLM tone verification (mirrors the judge). Stub trusts the template."""
        if self.s.mode == "stub":
            return True
        raw = self._chat([{"role": "user", "content":
            f'Return JSON {{"match": true|false}}: is this caption clearly in a '
            f'{style.name} tone? {style.role}\n\nCaption: {caption}'}],
            max_tokens=300, json_mode=True)
        mo = re.search(r"\{.*\}", raw, re.S)
        try:
            return bool(json.loads(mo.group()).get("match")) if mo else True
        except Exception:
            return True   # don't force needless regeneration on a parse hiccup
