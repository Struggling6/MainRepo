FROM flwr/superexec:1.27.0

USER root
RUN apt-get update \
    && apt-get -y --no-install-recommends install build-essential \
    && rm -rf /var/lib/apt/lists/*
USER app

WORKDIR /app

COPY --chown=app:app pyproject.toml .
COPY --chown=app:app . .

RUN python -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/rocm6.3

RUN python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]