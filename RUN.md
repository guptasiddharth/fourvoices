# Running FourVoices

Four ways to run, from zero-setup to production. The pipeline is identical across
all of them — only `LLM_BASE_URL` / `LLM_MODEL` change.

## Prerequisites
- **Python 3.11+** (core pipeline is stdlib-only).
- **ffmpeg** — frame sampling from real clips. `brew install ffmpeg` / `apt-get install ffmpeg`.
- Optional: **Ollama** (free local Gemma), **Docker**, a **Fireworks** API key.

Install the serving deps only if you want the HTTP API:
```bash
pip install -r requirements.txt      # fastapi, uvicorn, pydantic
```

---

## Mode 1 — Offline (no GPU, no keys)
Proves the whole pipeline with a deterministic stub model.
```bash
python3 eval/selfcheck.py        # 10/10: 4 styles present, distinct, faithful
python3 eval/test_rigorous.py    # 36/36: accuracy guard, distinctness, edge cases
```

## Mode 2 — Free local Gemma via Ollama  ★ recommended for dev/demo
Real Gemma weights, real vision, $0.
```bash
ollama pull gemma3:4b            # multimodal Gemma 3 (vision). gemma2:2b = text/styling only.
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_API_KEY=ollama
export LLM_MODEL=gemma3:4b

# One clip (folder of clips → one JSON each + a combined captions.json)
./scripts/run_batch.sh /path/to/clips outputs/
```

## Mode 3 — Gemma via Fireworks AI  (Track 2 access path / Gemma 4)
```bash
export LLM_BASE_URL=https://api.fireworks.ai/inference/v1
export LLM_API_KEY=fw-...                                   # rotate after the event
export LLM_MODEL=accounts/fireworks/models/gemma-4-26b-a4b-it   # needs a Fireworks "Dedicated" deployment
./scripts/run_batch.sh clips/ outputs/
```
> Note: Gemma 4 26B on Fireworks is not serverless — create a **Dedicated**
> deployment first (bills per GPU-hour → deploy, run, delete). For everyday work
> use Mode 2 (free).

## Mode 4 — Docker (containerized submission)  ✅ verified: builds, runs, `/health` + `/caption` respond
```bash
docker build -t fourvoices .          # ffmpeg included

# (a) Offline stub — proves it runs with no keys:
docker run -p 8000:8000 fourvoices

# (b) Real Gemma via Fireworks (public URL works directly):
docker run -p 8000:8000 \
  -e LLM_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e LLM_API_KEY=fw-... \
  -e LLM_MODEL=accounts/fireworks/models/gemma-4-26b-a4b-it \
  fourvoices

# (c) Free local Gemma via Ollama on the host — use host.docker.internal, NOT localhost:
docker run -p 8000:8000 \
  -e LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  -e LLM_API_KEY=ollama -e LLM_MODEL=gemma3:4b \
  fourvoices
```
Endpoints: `GET /health` · `POST /caption {"clip_path": "/data/clip.mp4"}` or `{"facts": "..."}`.
(To caption a local file, mount it: add `-v /path/clips:/data`.)

---

## HTTP API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
curl -s -X POST localhost:8000/caption -H 'content-type: application/json' \
  -d '{"facts":"a cat knocks a glass off a table in slow motion"}' | python3 -m json.tool
```
`/caption` returns `grounded_facts`, `captions` (the 4 voices), per-style
`checks` (accuracy/tone), and `distinct`.

## Reproduce the sample output
```bash
export LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=gemma3:4b
mkdir -p /tmp/clip && cp your_clip.mp4 /tmp/clip/
python3 -m app.batch /tmp/clip eval/sample_output/
```

## Config knobs (env)
| Var | Purpose | Default |
|---|---|---|
| `LLM_BASE_URL` | OpenAI-compatible endpoint (Ollama / Fireworks / vLLM) | — (stub) |
| `LLM_API_KEY` | key (`ollama` for local; `fw-…` for Fireworks) | `EMPTY` |
| `LLM_MODEL` | model id | `accounts/fireworks/models/gemma-4-26b-a4b-it` |
| `VC_N_FRAMES` | frames sampled per clip | `6` |
| `VC_LLM_MODE` | `auto` / `stub` / `openai` | `auto` |
