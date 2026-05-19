# syntax=docker/dockerfile:1
FROM python:3.12-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.1.4 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1 \
    SEVEN_ZIP_BIN=/usr/bin/7z \
    HOST=0.0.0.0 \
    PORT=1414

RUN apt-get update && apt-get install -y --no-install-recommends \
    p7zip-full \
    curl \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/7z /usr/local/bin/7z

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

COPY pyproject.toml poetry.lock ./
# [tool.poetry.dependencies] pins Windows CUDA wheels; not usable in Linux images
RUN sed -i '/^\[tool\.poetry\.dependencies\]/,$d' pyproject.toml \
    && poetry install --no-root --no-ansi

COPY docker/entrypoint.sh ./docker/entrypoint.sh
COPY api/ ./api/
COPY emb_fit/ ./emb_fit/
COPY parsers/ ./parsers/
COPY datasets/ ./datasets/
COPY app.py benchmark.py config.py embedding_qdrant.py tasks.py utils.py api.md ./

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p datasets/mun_data datasets/uploads logs emb_fit/checkpoints

EXPOSE 1414

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1414"]
