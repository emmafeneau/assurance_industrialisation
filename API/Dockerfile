FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copie les fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installe uniquement les dépendances API (pas les outils dev)
RUN uv sync --extra api --no-dev --frozen

# Copie le reste du projet
COPY API/app ./app
COPY Frequence/models ./Frequence/models
COPY Severite/models ./Severite/models

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]