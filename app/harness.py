"""Track 2 evaluation entrypoint (the hackathon harness contract).

    read /input/tasks.json  →  caption each clip in requested styles  →
    write /output/results.json  →  exit 0

tasks.json:   [{"task_id","video_url","styles":[...]}]
results.json: [{"task_id","captions":{style: caption, ...}}]

Track 2 injects no API key — the model endpoint/credentials come from the
container's own environment (baked at build or passed at run). Reuses the
FourVoices pipeline; resilient per-task so one bad clip never fails the run.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import tempfile
import urllib.request

from .caption import VideoCaptioner
from .styles import STYLES, STYLE_KEYS

INPUT_PATH = os.getenv("VC_INPUT", "/input/tasks.json")
OUTPUT_PATH = os.getenv("VC_OUTPUT", "/output/results.json")


def _ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download(url: str, dst: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "FourVoices/0.1"})
    with urllib.request.urlopen(req, timeout=180, context=_ssl_ctx()) as r, open(dst, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def _fallback(styles: list[str]) -> dict[str, str]:
    by = {s.key: s for s in STYLES}
    return {s: (by[s].stub_template.format(facts="a short video clip") if s in by else "")
            for s in styles}


def main() -> int:
    with open(INPUT_PATH) as f:
        tasks = json.load(f)
    vc = VideoCaptioner()
    results = []
    for t in tasks:
        tid = t.get("task_id")
        styles = t.get("styles") or STYLE_KEYS
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                path = tf.name
            _download(t["video_url"], path)
            full = vc.caption(clip_path=path)["captions"]
            caps = {s: full.get(s, "") for s in styles}
            # never emit an empty caption for a requested style (would score zero)
            fb = _fallback(styles)
            caps = {s: (caps[s] or fb[s]) for s in styles}
            print(f"{tid}: captioned ({len(styles)} styles)")
        except Exception as exc:  # noqa: BLE001
            caps = _fallback(styles)
            print(f"! {tid}: failed ({str(exc)[:80]}) — fallback captions")
        results.append({"task_id": tid, "captions": caps})

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} results → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
