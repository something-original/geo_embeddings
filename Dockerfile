# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# CPU: slim Python + stable CPU wheels, multi-stage
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder-cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.1.4 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml .

RUN poetry install --only main,cpu --no-root --no-ansi


FROM python:3.12-slim-bookworm AS runtime-cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SEVEN_ZIP_BIN=/usr/bin/7z \
    HOST=0.0.0.0 \
    PORT=1414 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    p7zip-full \
    curl \
    gdal-bin \
    libgdal32 \
    libgeos-c1v5 \
    libproj25 \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/7z /usr/local/bin/7z

COPY --from=builder-cpu /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder-cpu /usr/local/bin /usr/local/bin

WORKDIR /app
COPY docker/entrypoint.sh ./docker/entrypoint.sh
COPY api/ ./api/
COPY emb_fit/ ./emb_fit/
COPY parsers/ ./parsers/
COPY app.py config.py embedding_qdrant.py utils.py benchmark.py tasks.py ./

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p datasets/mun_data datasets/uploads logs emb_fit/checkpoints

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1414"]


# ---------------------------------------------------------------------------
# GPU: official PyTorch runtime (torch/torchvision already in the image)
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime AS runtime-gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.1.4 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    SEVEN_ZIP_BIN=/usr/bin/7z \
    HOST=0.0.0.0 \
    PORT=1414 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    p7zip-full \
    curl \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/7z /usr/local/bin/7z


RUN pip install --no-cache-dir --break-system-packages "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml .

RUN poetry install --only main --no-root --no-ansi

COPY docker/entrypoint.sh ./docker/entrypoint.sh
COPY api/ ./api/
COPY emb_fit/ ./emb_fit/
COPY parsers/ ./parsers/
COPY app.py config.py embedding_qdrant.py utils.py benchmark.py tasks.py ./

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p datasets/mun_data datasets/uploads logs emb_fit/checkpoints

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1414"]
