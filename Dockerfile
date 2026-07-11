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
# 26B MoE, vision). These are NOT secrets. Provide the one secret at run time:
#   docker run -e LLM_API_KEY=<your-google-ai-studio-key> -v ...:/input -v ...:/output <image>
# No key → the app runs in offline stub mode (valid output, generic captions).
ENV LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
    LLM_MODEL=gemma-4-26b-a4b-it \
    VC_N_FRAMES=12

# Track 2 evaluation contract: read /input/tasks.json → write /output/results.json → exit 0.
# (The FastAPI server + Streamlit demo are separate modules; override CMD to use them.)
CMD ["python", "-m", "app.harness"]
