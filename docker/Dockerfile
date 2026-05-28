# Reproducible training environment.
#
# Built on the official PyTorch CUDA base image so bitsandbytes / flash-attn
# wheels resolve correctly. Layered for build-cache friendliness:
#   1. Base image + system deps
#   2. pyproject + lock-equivalent (frozen pinned deps in pyproject.toml)
#   3. Project source last, so source edits do not invalidate the deps layer.

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer
COPY pyproject.toml README.md ./
RUN mkdir -p src/math_lora && touch src/math_lora/__init__.py \
    && pip install --upgrade pip \
    && pip install -e ".[dev]"

# Source layer
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY data ./data

# Volumes for outputs and HF cache so artifacts survive container restarts.
VOLUME ["/app/outputs", "/app/reports", "/cache/huggingface"]

CMD ["python", "-m", "math_lora.train", "--config", "configs/qwen-0.5b-lora.yaml"]
