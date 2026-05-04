FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY voice_agent/ ./voice_agent/

# Model cache — Piper voices + Whisper weights land here at prewarm.
# Mount as a named volume in docker-compose so models survive redeploys.
ENV LOCAL_MODEL_CACHE=/root/.cache/nexuscaller-local-models

EXPOSE 8765

HEALTHCHECK CMD curl --fail http://localhost:8765/health || exit 1

# Default: run the FastAPI server.
# Override in docker-compose with ["python", "-m", "voice_agent.agent", "start"]
# for the LiveKit agent worker process.
CMD ["uvicorn", "voice_agent.server:app", "--host", "0.0.0.0", "--port", "8765"]
