FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODE_DATA_DIR=/app/data \
    PORT=8080 \
    OLARES_CLI=/usr/local/bin/olares-cli

WORKDIR /srv

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g @olares/cli@1.12.8-cli.0 \
 && OCLI="$(find /usr/lib /usr/local -name olares-cli -type f 2>/dev/null | head -1)" \
 && if [ -n "$OCLI" ]; then ln -sf "$OCLI" /usr/local/bin/olares-cli; fi \
 && olares-cli version \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/server.py ./
COPY app/static ./static/

RUN useradd --uid 1000 --create-home app \
 && mkdir -p /app/data \
 && chown -R app:app /app/data /srv

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"

CMD ["python", "server.py"]
