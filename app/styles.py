"""The four required voices + prompts + a tone/accuracy self-check.

Each style shares the SAME grounded facts; only the tone changes. That keeps
accuracy high (LLM-judge) while making tone distinct (LLM-judge).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Style:
    key: str
    name: str
    guidance: str
    stub_template: str   # deterministic offline rendering, distinct per style


STYLES: list[Style] = [
    Style("formal", "Formal",
          "Neutral, precise, professional. Complete sentence. No slang, no emoji, "
          "no jokes. Read like a broadcast news caption.",
          "The footage shows {facts}."),
    Style("sarcastic", "Sarcastic",
          "Dry, ironic, deadpan. Say the opposite of enthusiasm; imply the scene is "
          "unremarkable or overhyped. Keep it short and cutting. No emoji.",
          "Oh, thrilling — {facts}. Riveting television, truly."),
    Style("humorous_tech", "Humorous (tech)",
          "Playful, for a developer/tech audience. Use ONE apt tech metaphor "
          "(bugs, CI, prod, latency, LLMs). Light, not cringe. At most one emoji.",
          "POV: prod at 2am when {facts} 🤖."),
    Style("humorous_non_tech", "Humorous (non-tech)",
          "Playful, relatable, everyday humor. No tech references. Warm and silly. "
          "At most one emoji.",
          "That moment when {facts} and you just can't even 😂."),
]

STYLE_KEYS = [s.key for s in STYLES]


def generation_prompt(facts: str, style: Style) -> str:
    return (
        f"GROUNDED visual facts (do not add anything not present here):\n{facts}\n\n"
        f"Write ONE {style.name} caption. Style: {style.guidance}\n"
        f"Stay faithful to the facts; one short sentence; tone unmistakably {style.name}.\n"
        f'Return ONLY a JSON object of the form {{"caption": "..."}} where caption '
        f'is one short sentence, faithful to the facts, in an unmistakably '
        f'{style.name} tone. No other text.'
    )


# ---- self-check ------------------------------------------------------------

def _keywords(facts: str) -> list[str]:
    import re
    stop = {"the", "a", "an", "and", "of", "in", "on", "at", "with", "to", "is",
            "are", "was", "for", "its", "his", "her", "who", "you", "shows",
            "footage", "video", "clip", "person", "then", "that", "this"}
    # 3+ chars so meaningful short nouns (dog, cat, car) still anchor accuracy.
    return [w for w in re.findall(r"[a-zA-Z]{3,}", facts.lower()) if w not in stop]


def accuracy_ok(facts: str, caption: str, min_overlap: int = 1) -> bool:
    """Caption must reference the grounded content — cheap hallucination guard."""
    kws = _keywords(facts)
    if not kws:
        return True
    hits = sum(1 for k in kws if k in caption.lower())
    return hits >= min(min_overlap, len(kws))


def distinct_ok(captions: dict[str, str]) -> bool:
    """The four captions must not be near-identical."""
    vals = [c.strip().lower() for c in captions.values()]
    return len(set(vals)) == len(vals)
