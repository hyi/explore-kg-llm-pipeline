FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8050 \
    HOME=/tmp \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY explorer ./explorer
COPY src ./src

EXPOSE 8050

CMD ["gunicorn", "explorer.frontend.dash_app:server", "--bind", "0.0.0.0:8050", "--workers", "1", "--threads", "4", "--timeout", "180"]
