# Running FourVoices

The pipeline is identical everywhere — only three env vars change where Gemma runs:
`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. With no endpoint set it runs offline
in a deterministic stub mode (valid output, no keys, no GPU).

## Prerequisites
- **Python 3.11+** (core pipeline is stdlib-only).
- **ffmpeg** — frame sampling from real clips. `brew install ffmpeg` / `apt-get install ffmpeg`.
- A **Gemma endpoint**: a free [Google AI Studio](https://aistudio.google.com/apikey) key
  (validated default) *or* local [Ollama](https://ollama.com) (free) *or* any
  OpenAI-compatible Gemma host (vLLM on MI300X, Fireworks, …).

```bash
pip install -r requirements.txt
```

---

## Track 2 evaluation contract  ✅ this is what the grader runs
The container reads `/input/tasks.json`, downloads each clip, captions it in the
requested styles, writes `/output/results.json`, and exits 0.

```jsonc
// /input/tasks.json
[{"task_id": "v1", "video_url": "https://…/clip.mp4",
  "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}]
```
```jsonc
// /output/results.json
[{"task_id": "v1", "captions": {"formal": "…", "sarcastic": "…",
  "humorous_tech": "…", "humorous_non_tech": "…"}}]
```

### Run the graded container (Gemma 4 on Google's free API — validated)
The image is prebuilt for **linux/amd64** and pre-pointed at Gemma 4. Supply only
the key:
```bash
docker run --rm \
  -e LLM_API_KEY=<your-google-ai-studio-key> \
  -v "$PWD/input:/input" -v "$PWD/output:/output" \
  ghcr.io/guptasiddharth/fourvoices:latest
```
No key → it runs in offline stub mode (still valid JSON, exit 0). Baked-in
non-secret defaults: `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`,
`LLM_MODEL=gemma-4-26b-a4b-it`.

### Run the harness locally (no Docker)
```bash
export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export LLM_API_KEY=<your-google-ai-studio-key>
export LLM_MODEL=gemma-4-26b-a4b-it
VC_INPUT=input/tasks.json VC_OUTPUT=output/results.json python3 -m app.harness
```

---

## Offline (no GPU, no keys) — prove the pipeline
```bash
python3 eval/selfcheck.py        # 4 styles present, distinct, faithful
python3 eval/test_rigorous.py    # accuracy guard, distinctness, edge cases
```

## Free local Gemma via Ollama  ★ zero-cost alternative
```bash
ollama pull gemma3:4b            # multimodal Gemma (vision). gemma2:2b = text/styling only.
export LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=gemma3:4b
# in Docker, use http://host.docker.internal:11434/v1 instead of localhost
```

## Demo UI / HTTP API (separate entrypoints; override the container CMD)
```bash
streamlit run streamlit_app.py                       # live demo UI
uvicorn app.api:app --host 0.0.0.0 --port 8000       # POST /caption, GET /health
```

---

## Config knobs (env)
| Var | Purpose | Default |
|---|---|---|
| `LLM_BASE_URL` | OpenAI-compatible Gemma endpoint | Google AI Studio (in image) |
| `LLM_API_KEY` | key (`ollama` for local; AI-Studio key for Google) | `EMPTY` → stub mode |
| `LLM_MODEL` | Gemma model id | `gemma-4-26b-a4b-it` |
| `VC_N_FRAMES` | frames sampled per clip (scales up with clip length) | `8` (image sets `4`) |
| `VC_LLM_MODE` | `auto` / `stub` / `openai` | `auto` |
| `VC_INPUT` / `VC_OUTPUT` | harness I/O paths | `/input/tasks.json`, `/output/results.json` |
