FROM flwr/superexec:1.27.0

USER root
RUN apt-get update \
    && apt-get -y --no-install-recommends install build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -f render \
    && groupadd -f video \
    && usermod -aG render,video app \
    && ln -sf /python/venv/bin/python3 /usr/local/bin/python3 \
    && ln -sf /python/venv/bin/python3 /usr/local/bin/python

# Install uv into the venv
RUN /python/venv/bin/pip install uv

ENV PATH="/python/venv/bin:$PATH"
ENV PYTHONPATH="/python/venv/lib/python3.13/site-packages:/app"
ENV MIOPEN_USER_DB_PATH="/tmp/miopen"
ENV MIOPEN_SYSTEM_DB_PATH="/tmp/miopen"
ENV MIOPEN_DISABLE_CACHE=1

ARG TORCH_VARIANT=cpu
ARG TORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.3
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN case "${TORCH_VARIANT}" in \
        cpu)  uv pip install --python /python/venv/bin/python torch torchvision ;; \
        cuda) uv pip install --python /python/venv/bin/python torch torchvision --index-url "${TORCH_CUDA_INDEX_URL}" ;; \
        rocm) uv pip install --python /python/venv/bin/python torch torchvision --index-url "${TORCH_ROCM_INDEX_URL}" ;; \
        *) echo "Unsupported TORCH_VARIANT: ${TORCH_VARIANT}" >&2; exit 1 ;; \
    esac


WORKDIR /app
COPY --chown=app:app . .
RUN uv pip install --python /python/venv/bin/python .
USER app

ENTRYPOINT ["flower-superexec"]