#!/usr/bin/env bash
# Caption every clip in a folder → JSON per clip.
#   ./scripts/run_batch.sh <clips_dir> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m app.batch "${1:-clips}" "${2:-outputs}"
