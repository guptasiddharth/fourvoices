"""Offline self-check (stub Gemma — no GPU, no keys).

Proves the pipeline emits all four required styles, keeps them faithful to the
grounded facts (accuracy), and keeps them distinct (tone) — the two axes the
hackathon LLM-judge scores.

Run from project root:  python3 eval/selfcheck.py
"""

from __future__ import annotations

import os
import sys

os.environ["VC_LLM_MODE"] = "stub"
os.environ.pop("LLM_BASE_URL", None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.caption import VideoCaptioner            # noqa: E402
from app.styles import STYLE_KEYS, accuracy_ok, distinct_ok  # noqa: E402

REQUIRED = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


def main() -> int:
    p = t = 0
    vc = VideoCaptioner()

    for label, facts in [("stub-grounded clip", None),
                         ("explicit facts", "a chef flips a pancake that lands on the ceiling")]:
        r = vc.caption(facts=facts)
        caps = r["captions"]
        grounded = r["grounded_facts"]

        for check, ok in [
            ("all four styles present", set(caps) == set(REQUIRED) == set(STYLE_KEYS)),
            ("all captions non-empty", all(c.strip() for c in caps.values())),
            ("captions distinct (tone varies)", distinct_ok(caps)),
            ("every caption faithful to facts", all(accuracy_ok(grounded, c) for c in caps.values())),
            ("self-check ran per style", set(r["checks"]) == set(REQUIRED)),
        ]:
            t += 1
            p += ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {check}")

        print(f"    facts: {grounded}")
        for k in REQUIRED:
            print(f"      {k:18} {caps[k]}")
        print()

    print(f"FourVoices self-check: {p}/{t} = {p / t:.0%}")
    return 0 if p == t else 1


if __name__ == "__main__":
    raise SystemExit(main())
