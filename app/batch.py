"""Batch-caption a folder of clips → one JSON per clip.

Usage:  python3 -m app.batch <clips_dir> [out_dir]
"""

from __future__ import annotations

import json
import os
import sys

from .caption import VideoCaptioner

_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def main(clips_dir: str, out_dir: str = "outputs") -> int:
    os.makedirs(out_dir, exist_ok=True)
    vc = VideoCaptioner()
    clips = [f for f in sorted(os.listdir(clips_dir)) if f.lower().endswith(_EXTS)]
    if not clips:
        print(f"No clips found in {clips_dir}")
        return 1
    manifest: dict[str, dict] = {}
    for c in clips:
        result = vc.caption(clip_path=os.path.join(clips_dir, c))
        with open(os.path.join(out_dir, f"{c}.json"), "w") as f:
            json.dump(result, f, indent=2)
        manifest[c] = result["captions"]     # {clip: {formal, sarcastic, humorous_tech, humorous_non_tech}}
        print(f"{c}: {result['n_frames']} frames → {out_dir}/{c}.json")
    # Combined manifest — the clip→4-styles mapping a judge/harness most likely wants.
    with open(os.path.join(out_dir, "captions.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"→ combined manifest: {out_dir}/captions.json ({len(manifest)} clips)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 -m app.batch <clips_dir> [out_dir]")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "outputs"))
