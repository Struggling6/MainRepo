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
# Avoid warnings about missing AMD GPU IDs file
ENV AMDGPU_IDS_PATH="/dev/null"

# requirements.txt is generated before build via:
#   python generate_requirements.py --variant <cpu|cuda|rocm>
COPY requirements.txt .
RUN uv pip install --python /python/venv/bin/python -r requirements.txt

WORKDIR /app
COPY --chown=app:app . .
RUN uv pip install --python /python/venv/bin/python --no-deps .
USER app

ENTRYPOINT ["flower-superexec"]