FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra api --no-dev --frozen

COPY api/app ./app
COPY Frequence/models ./Frequence/models
COPY Frequence/src ./Frequence/src
COPY Severite/models ./Severite/models
COPY Severite/src ./Severite/src

ENV PATH="/app/.venv/bin:$PATH"
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}