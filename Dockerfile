FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Optional extra root CAs (e.g. a corporate TLS-inspecting proxy): any *.crt
# in docker/certs/ is trusted by the system, pip and uv. No-op when empty.
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# LightGBM needs the OpenMP runtime, absent from the slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Install dependencies first (better layer caching on code-only changes).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY models ./models
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
