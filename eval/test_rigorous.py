"""Rigorous / adversarial test suite for FourVoices.

Goal: try to BREAK the two claims the LLM-judge scores — (1) every caption stays
faithful to the grounded facts (accuracy: no hallucinated content), and (2) the
four voices are present and distinct (tone). Runs offline (stub Gemma).

    python3 eval/test_rigorous.py
"""

from __future__ import annotations

import os
import sys

os.environ["VC_LLM_MODE"] = "stub"
os.environ.pop("LLM_BASE_URL", None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.caption import VideoCaptioner                       # noqa: E402
from app.styles import (STYLE_KEYS, accuracy_ok, distinct_ok,  # noqa: E402
                        generation_prompt, STYLES)

REQUIRED = {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}
_P = _T = 0
_FAILS: list[str] = []


def check(name: str, cond: bool) -> None:
    global _P, _T
    _T += 1
    _P += bool(cond)
    if not cond:
        _FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ---------------------------------------------------- accuracy guard (anti-hallucination)
print("== accuracy guard rejects hallucinated captions ==")
facts = "a chef flips a pancake that lands on the ceiling"
check("faithful caption accepted", accuracy_ok(facts, "The chef's pancake hit the ceiling."))
check("unrelated caption rejected", accuracy_ok(facts, "A spaceship launches toward Mars.") is False)
check("partial-overlap accepted", accuracy_ok(facts, "Nice pancake."))
check("short-noun facts still checked (dog)",
      accuracy_ok("a dog runs", "a dog runs fast") and not accuracy_ok("a dog runs", "a bird flies"))
check("empty facts → vacuously true", accuracy_ok("", "anything"))

# ---------------------------------------------------- distinctness
print("== distinctness ==")
check("identical captions → not distinct",
      distinct_ok({"a": "x", "b": "x", "c": "y", "d": "z"}) is False)
check("all-different → distinct",
      distinct_ok({"a": "1", "b": "2", "c": "3", "d": "4"}))

# ---------------------------------------------------- style config sanity
print("== style config ==")
check("exactly the 4 required styles", set(STYLE_KEYS) == REQUIRED)
check("generation_prompt embeds facts + style",
      "chef" in generation_prompt("a chef cooks", STYLES[0]).lower())

# ---------------------------------------------------- pipeline across many inputs
print("== pipeline across varied clips (accuracy + tone + distinct) ==")
CLIPS = [
    "a person waves at a small dog running across a sunlit park",
    "a chef flips a pancake that lands on the ceiling",
    "two engineers debug a server while alarms flash red",
    "a toddler stacks blocks then knocks the tower down laughing",
    "a cyclist splashes through a puddle on a rainy street",
]
vc = VideoCaptioner()
for facts in CLIPS:
    r = vc.caption(facts=facts)
    caps = r["captions"]
    check(f"[{facts[:28]}…] 4 styles present", set(caps) == REQUIRED)
    check(f"[{facts[:28]}…] all non-empty", all(c.strip() for c in caps.values()))
    check(f"[{facts[:28]}…] distinct voices", distinct_ok(caps))
    check(f"[{facts[:28]}…] every voice faithful", all(accuracy_ok(facts, c) for c in caps.values()))
    check(f"[{facts[:28]}…] checks recorded per style", set(r["checks"]) == REQUIRED)

# ---------------------------------------------------- edge cases
print("== edge cases ==")
r_short = vc.caption(facts="a cat sleeps")
check("short facts → still 4 distinct styles",
      set(r_short["captions"]) == REQUIRED and distinct_ok(r_short["captions"]))
r_num = vc.caption(facts="a runner crosses the finish line at 9.58 seconds")
check("numeric facts preserved in formal caption", "9.58" in r_num["captions"]["formal"])

print(f"\nRIGOROUS: {_P}/{_T} = {_P / _T:.0%}")
if _FAILS:
    print("FAILURES:", _FAILS)
raise SystemExit(0 if _P == _T else 1)
