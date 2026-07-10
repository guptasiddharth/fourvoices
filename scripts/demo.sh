#!/usr/bin/env bash
# Recording-friendly demo: caption ONE clip with Gemma and pretty-print the four
# voices. Defaults to free local Gemma via Ollama; override LLM_* for Fireworks.
#   ./scripts/demo.sh /path/to/clip.mp4
set -euo pipefail
cd "$(dirname "$0")/.."
CLIP="${1:?usage: ./scripts/demo.sh <clip.mp4>}"
: "${LLM_BASE_URL:=http://localhost:11434/v1}"
: "${LLM_API_KEY:=ollama}"
: "${LLM_MODEL:=gemma3:4b}"
: "${VC_N_FRAMES:=4}"
export LLM_BASE_URL LLM_API_KEY LLM_MODEL VC_N_FRAMES VC_LLM_MODE=openai
python3 - "$CLIP" <<'PY'
import os, sys, time
sys.path.insert(0, ".")
from app.caption import VideoCaptioner
clip = sys.argv[1]
print(f"\n  FourVoices  —  grounded four-style video captioning with Gemma\n")
print(f"  clip   {clip}")
print(f"  model  {os.environ['LLM_MODEL']}  (via {os.environ['LLM_BASE_URL']})\n")
t = time.time()
r = VideoCaptioner().caption(clip_path=clip)
print(f"  Gemma vision (grounded facts):")
print(f"      {r['grounded_facts']}\n")
labels = {"formal": "FORMAL", "sarcastic": "SARCASTIC",
          "humorous_tech": "HUMOROUS · TECH", "humorous_non_tech": "HUMOROUS · NON-TECH"}
for k, v in r["captions"].items():
    print(f"    {labels[k]:20}  {v}")
print(f"\n  {r['n_frames']} frames · faithful + distinct · {time.time()-t:.0f}s · $0 (local Gemma)\n")
PY
