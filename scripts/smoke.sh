#!/usr/bin/env bash
# Offline proof (no GPU/keys): self-check, then the HTTP API if fastapi is present.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== self-check (stub Gemma) ==="
VC_LLM_MODE=stub python3 eval/selfcheck.py

if python3 -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "=== API smoke ==="
  VC_LLM_MODE=stub python3 -m uvicorn app.api:app --port 8078 &
  PID=$!; sleep 3
  curl -s localhost:8078/health; echo
  curl -s -X POST localhost:8078/caption -H 'content-type: application/json' \
    -d '{"facts":"a cat knocks a glass off a table in slow motion"}' | python3 -m json.tool
  kill $PID 2>/dev/null || true
else
  echo "(pip install -r requirements.txt to also smoke-test the HTTP API)"
fi
