# FourVoices video captioner (submission requirement: containerized).
# ffmpeg is installed for frame sampling; Gemma inference is remote (Fireworks)
# or a vLLM Gemma server — no model weights baked in.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY eval ./eval

# Pre-pointed at Fireworks AI (qwen3p7-plus multimodal, reasoning disabled for clean
# captions; kimi-k2p6 as a vision-capable fallback). The eval platform provides no
# runtime env field, so the API key is baked at BUILD time and the container is
# self-sufficient:
#   docker build --build-arg LLM_API_KEY=<your-fireworks-key> ...
# The key is NOT stored in this repo — it lives only inside the built image.
# Rotate it after judging. No key baked → offline stub mode.
# (Model-agnostic: override LLM_BASE_URL/LLM_MODEL/LLM_REASONING_EFFORT for any
#  OpenAI-compatible host, e.g. Gemma 4 on Google AI Studio.)
ARG LLM_API_KEY=""
ENV LLM_BASE_URL=https://api.fireworks.ai/inference/v1 \
    LLM_MODEL=accounts/fireworks/models/qwen3p7-plus \
    LLM_MODEL_FALLBACK=accounts/fireworks/models/kimi-k2p6 \
    LLM_REASONING_EFFORT=none \
    VC_N_FRAMES=8 \
    LLM_API_KEY=${LLM_API_KEY}

# Track 2 evaluation contract: read /input/tasks.json → write /output/results.json → exit 0.
# (The FastAPI server + Streamlit demo are separate modules; override CMD to use them.)
CMD ["python", "-m", "app.harness"]
