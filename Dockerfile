# Reproduction image: pinned dependencies from uv.lock, Python 3.12, CUDA runtime for vLLM/torch.
# Build:  docker build -t advbug .
# Run:    docker run --gpus all -v $PWD/data:/app/data -v $PWD/runs:/app/runs advbug \
#             python scripts/run_queue.py --tier 1 --seeds 0 1 2
# Data is not baked in: rebuild it with scripts/build_*.py or mount data/ (manifests record the hashes).
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 && uv sync --frozen --no-install-project --all-extras
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY PLAN.md PROGRESS.md ./
RUN uv sync --frozen --all-extras

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
CMD ["python", "scripts/train.py", "--arm", "configs/arms/full.yaml"]
