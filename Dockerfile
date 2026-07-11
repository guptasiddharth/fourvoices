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

# Pre-pointed at Gemma 4 on Google's free OpenAI-compatible endpoint (validated:
# 26B MoE, vision). The eval platform provides no runtime env field, so the API key
# is baked at BUILD time and the container is self-sufficient:
#   docker build --build-arg LLM_API_KEY=<your-google-ai-studio-key> ...
# The key is NOT stored in this repo — it lives only inside the built image. Use a
# free-tier key and rotate it after judging. No key baked → offline stub mode.
ARG LLM_API_KEY=""
ENV LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
    LLM_MODEL=gemma-4-26b-a4b-it \
    VC_N_FRAMES=12 \
    LLM_API_KEY=${LLM_API_KEY}

# Track 2 evaluation contract: read /input/tasks.json → write /output/results.json → exit 0.
# (The FastAPI server + Streamlit demo are separate modules; override CMD to use them.)
CMD ["python", "-m", "app.harness"]
