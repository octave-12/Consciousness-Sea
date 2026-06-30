FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY pyproject.toml .
COPY backend/ backend/
RUN pip install --no-cache-dir -e .

FROM python:3.12-slim AS runtime

WORKDIR /app

RUN groupadd -r consciousness && useradd -r -g consciousness consciousness

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/consciousness-sea /usr/local/bin/consciousness-sea
COPY --from=builder /app /app

RUN mkdir -p /app/data /app/checkpoints /app/certs && chown -R consciousness:consciousness /app/data /app/checkpoints /app/certs

USER consciousness

ENV CONSCIOUSNESS_SEA_DATA_DIR=/app/data
ENV CONSCIOUSNESS_SEA_DB_PATH=/app/data/consciousness_sea.db

EXPOSE 8111

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8111/health')" || exit 1

CMD ["consciousness-sea"]
