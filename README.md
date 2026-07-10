# FourVoices — grounded four-style video captioning with Gemma

> **AMD Developer Hackathon: Act II · Track 2 (Video Captioning)** · targeting **Best Use of Gemma in Video Captioning**.

Caption any short clip (30 s–2 min) in **four distinct voices** — *formal,
sarcastic, humorous-tech, humorous-non-tech* — using **Gemma's multimodal
vision**. Judged on **accuracy + tone**, and FourVoices is built to win on both.

## The idea: **ground once, style four times**
Video captioning is scored on two things that fight each other — *accuracy* and
*tone*. Push a funny voice and models start inventing things that aren't in the
video; stay accurate and the tone goes flat. FourVoices decouples them:

```
clip ─► ffmpeg frames ─► Gemma vision ─► GROUNDED facts ─► ┌ formal
                          (what's actually on screen)      ├ sarcastic
                                                           ├ humorous-tech
                                                           └ humorous-non-tech
                                             └─► accuracy + tone self-check ─► regenerate drifters
```

All four voices are generated **from the same grounded description**, so content
stays faithful while only the tone changes. A self-check verifies each caption on
the exact axes the judge scores and regenerates any that drift.

## Real output (your call — this is genuine)
Clip: *dancing cats* · Model: `gemma3:4b` via Ollama (**local, free**) · 4 frames sampled.

**Gemma vision (grounded facts):** *"Five cartoon cats with pink sunglasses are standing in a row. Each cat is holding its arms up and appears to be dancing."*

| Voice | Caption |
|---|---|
| Formal | Five cartoon cats with pink sunglasses are depicted dancing while holding their arms upwards. |
| Sarcastic | They're certainly moving. |
| Humorous-tech | These cats are debugging our CI pipeline with some serious latency. |
| Humorous-non-tech | Someone get these cats a disco! |

All four faithful, all distinct — end-to-end in ~38 s at **$0**. See `eval/sample_output/`.

## Quickstart

**Offline (no GPU, no keys) — proves the pipeline:**
```bash
python3 eval/selfcheck.py        # 10/10
python3 eval/test_rigorous.py    # 36/36 adversarial (accuracy guard, distinctness, edge cases)
```

**Free local Gemma (recommended for dev/demo) — real Gemma, $0:**
```bash
ollama pull gemma3:4b            # multimodal Gemma (vision)
export LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=gemma3:4b
./scripts/run_batch.sh /path/to/clips outputs/     # one JSON of 4 captions per clip + captions.json
```

**Gemma via Fireworks AI (Track 2 access path / Gemma 4):**
```bash
export LLM_BASE_URL=https://api.fireworks.ai/inference/v1 LLM_API_KEY=fw-... \
       LLM_MODEL=accounts/fireworks/models/gemma-4-26b-a4b-it
./scripts/run_batch.sh clips/ outputs/
```

**Containerized:**
```bash
docker build -t fourvoices . && docker run -p 8000:8000 fourvoices   # ffmpeg baked in
# POST /caption {clip_path|facts} · GET /health
```

The model is a single env var (`LLM_MODEL`/`LLM_BASE_URL`) — the same code runs
on local Ollama Gemma, Fireworks Gemma 4, or Gemma on AMD Developer Cloud.

## Why it wins on accuracy + tone
- **Accuracy:** captions are constrained to Gemma's grounded description; a
  hallucination guard rejects any caption that drifts off-content.
- **Tone:** per-voice prompts + an LLM tone-check, with the four enforced distinct.
- **Robustness:** structured **JSON output** keeps captions clean even on
  reasoning-heavy models (no preamble leakage).

## Layout
```
app/  config · llm (stub↔Gemma, multimodal, JSON output) · frames (ffmpeg) ·
      styles (4 voices + accuracy/tone checks) · caption · batch · api
eval/ selfcheck.py · test_rigorous.py · sample_output/ (real Gemma demo)
scripts/ run_batch.sh · smoke.sh
Dockerfile · docker-compose (n/a) · requirements.txt · .env.example · LICENSE (MIT) · RUN.md · SUBMISSION.md
```

## Tech
Gemma (multimodal) · Fireworks AI · AMD Developer Cloud / ROCm · Ollama (free local
dev) · ffmpeg · Python / FastAPI · Docker. MIT-licensed.

See **RUN.md** for full run modes and **SUBMISSION.md** for the hackathon writeup.
