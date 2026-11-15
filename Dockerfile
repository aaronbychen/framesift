FROM python:3.12-slim

# gcc: triton (pulled in by torch/bitsandbytes) compiles a small CUDA helper at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY scanner/ scanner/
COPY templates/ templates/

# Model weights are mounted read-only at /app/.data/models (see docker-compose.yml);
# nothing is downloaded at runtime.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    REDIS_URL=redis://redis:6379 \
    UPLOAD_DIR=/app/uploads FRAMES_DIR=/app/frames
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
