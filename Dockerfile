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

# Track 2 evaluation contract: read /input/tasks.json → write /output/results.json → exit 0.
# (The FastAPI server + Streamlit demo are separate modules; override CMD to use them.)
CMD ["python", "-m", "app.harness"]
