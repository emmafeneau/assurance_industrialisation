FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra api --no-dev --frozen

COPY api/app ./app
COPY frequence/models ./frequence/models
COPY frequence/src ./frequence/src
COPY severite/models ./severite/models
COPY severite/src ./severite/src

ENV PATH="/app/.venv/bin:$PATH"
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}