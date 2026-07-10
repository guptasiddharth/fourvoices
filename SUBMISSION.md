# FourVoices — AMD Developer Hackathon: Act II

**Track 2 (Video Captioning) · targeting "Best Use of Gemma in Video Captioning" ($3,000)**

**Grounded four-style video captioning: ground once with Gemma's vision, style four times.**

## The task
Given short clips (30s–2min), produce a caption/summary in four distinct styles —
formal, sarcastic, humorous-tech, humorous-non-tech. Scored by an **LLM-judge on
accuracy and tone**.

## The insight
Accuracy and tone pull against each other: crank the humor and models start
inventing content that isn't in the video. FourVoices **decouples** them:

1. **Ground once** — Gemma's multimodal vision reads sampled frames and emits
   neutral, faithful *visual facts* (objects, actions, setting; no opinion).
2. **Style four times** — each voice is generated *from those same facts*, so
   content stays accurate while tone changes.
3. **Self-check** — every caption is verified for **accuracy** (faithful to the
   grounded facts) and **tone** (matches its target voice); drifters are
   regenerated. We optimize exactly the two axes the judge scores.

```
clip → ffmpeg frames → Gemma (grounded facts) → {formal, sarcastic, humorous-tech, humorous-non-tech} → accuracy+tone self-check
```

## Best use of Gemma
- **Gemma multimodal vision** is the grounding engine — it *watches* the frames.
- **Gemma** generates and self-verifies all four voices.
- Runs on AMD: Gemma via **Fireworks AI** (Track 2 access) or a **vLLM Gemma
  server on an MI300X** — same code, one env var.

## Why it wins on accuracy + tone
- **Accuracy:** captions are constrained to grounded facts; a hallucination
  guard rejects captions that drift off-content.
- **Tone:** style-specific prompts + exemplars + an LLM tone-check make each
  voice unmistakable, and the four are enforced to be distinct.

## Results (offline, reproducible — no GPU/keys)
`eval/selfcheck.py` → **10/10**: all four styles present, non-empty, mutually
distinct, and each faithful to the grounded facts, with the per-style self-check
running. Swapping in real Gemma is a `LLM_BASE_URL` change.

## How the AMD/Gemma credits are used
- **Fireworks AI ($50):** call Gemma multimodal for grounding + styling.
- **AMD Developer Cloud ($100, MI300X):** optionally self-host Gemma via vLLM, or
  fine-tune a captioner (explicitly allowed) on-device — same endpoint contract.

## Submission checklist
- [x] Containerized — `Dockerfile` (ffmpeg included).
- [x] MIT licensed — `LICENSE`.
- [x] Public GitHub repo + README with setup/usage.
- [x] Runnable via provided instructions (`eval/selfcheck.py`, `scripts/run_batch.sh`).
- [ ] Demo video · slides · deployed URL — _to record_.
- [ ] Team name / repo URL — _fill in_.

## Reproduce
```bash
python3 eval/selfcheck.py                    # 10/10, no GPU/keys
./scripts/run_batch.sh clips/ outputs/       # real clips once LLM_BASE_URL is set
```
