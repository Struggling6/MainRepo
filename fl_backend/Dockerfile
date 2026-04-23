FROM flwr/superexec:1.27.0

USER root
RUN apt-get update \
    && apt-get -y --no-install-recommends install build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -f render \
    && groupadd -f video \
    && usermod -aG render,video app
USER app

WORKDIR /app

# ── Install PyTorch first so this layer is cached across code changes ── #
ARG TORCH_VARIANT=cpu
ARG TORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.3
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN case "${TORCH_VARIANT}" in \
        cpu)  python -m pip install --no-cache-dir torch torchvision ;; \
        cuda) python -m pip install --no-cache-dir torch torchvision --index-url "${TORCH_CUDA_INDEX_URL}" ;; \
        rocm) python -m pip install --no-cache-dir torch torchvision --index-url "${TORCH_ROCM_INDEX_URL}" ;; \
        *) echo "Unsupported TORCH_VARIANT: ${TORCH_VARIANT}" >&2; exit 1 ;; \
    esac

# ── Install other dependencies before copying source ─────────────────── #
# Copying pyproject.toml alone lets pip install deps without invalidating
# the layer when only source files change.
COPY --chown=app:app pyproject.toml .
RUN python -m pip install --no-cache-dir --no-deps .

# ── Copy source last — only this layer rebuilds on code changes ───────── #
COPY --chown=app:app . .

ENTRYPOINT ["flower-superexec"]