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

COPY --chown=app:app pyproject.toml .
COPY --chown=app:app . .

ARG TORCH_VARIANT=cpu
ARG TORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.3
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN case "${TORCH_VARIANT}" in \
        cpu)  python -m pip install --no-cache-dir torch torchvision ;; \
        cuda) python -m pip install --no-cache-dir torch torchvision --index-url "${TORCH_CUDA_INDEX_URL}" ;; \
        rocm) python -m pip install --no-cache-dir torch torchvision --index-url "${TORCH_ROCM_INDEX_URL}" ;; \
        *) echo "Unsupported TORCH_VARIANT: ${TORCH_VARIANT}" >&2; exit 1 ;; \
    esac

RUN python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]