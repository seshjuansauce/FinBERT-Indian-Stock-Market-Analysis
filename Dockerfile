# Reproducible CPU environment for the whole pipeline.
#   docker compose build
#   docker compose run --rm finbert            # full reproduction (see scripts/reproduce.sh)
#   docker compose run --rm finbert prices     # a single stage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image ~1 GB smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .
ENTRYPOINT ["bash", "scripts/reproduce.sh"]
CMD ["all"]
