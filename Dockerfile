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

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
