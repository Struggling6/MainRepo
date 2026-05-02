FROM flwr/superexec:1.27.0

USER root
RUN apt-get update \
    && apt-get -y --no-install-recommends install build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -f render \
    && groupadd -f video \
    && usermod -aG render,video app \
    && ln -sf /python/venv/bin/python3 /usr/local/bin/python3

# NOTE: We intentionally do NOT symlink /usr/local/bin/python to the venv
# because the base image ships its system Python tree at /usr/local/bin/python/
# (yes, it's a directory). `ln -sf` against an existing directory silently
# places the link inside it, which is a no-op. Use /python/venv/bin/python or
# /usr/local/bin/python3 explicitly.

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
# python generate_requirements.py --variant <cpu|cuda|rocm>
COPY requirements.txt .
RUN uv pip install --python /python/venv/bin/python --index-strategy unsafe-best-match -r requirements.txt

# Smoke-test heavy native deps so a partial/broken install fails the build
# instead of silently producing an image where `import torch` fails at runtime.
RUN /python/venv/bin/python - <<'PY'
import importlib, sys
required = ["torch", "numpy", "pandas", "sklearn", "flwr", "transformers", "captum", "matplotlib"]
missing = []
for name in required:
    try:
        m = importlib.import_module(name)
        print(f"[smoke] OK  {name:<14} {getattr(m, '__version__', '?')} -> {getattr(m, '__file__', '?')}")
    except Exception as e:
        missing.append((name, repr(e)))
        print(f"[smoke] FAIL {name}: {e}")
if missing:
    sys.exit(f"Smoke test failed: {missing}")
PY

WORKDIR /app
COPY --chown=app:app . .
RUN uv pip install --python /python/venv/bin/python --no-deps .
USER app

ENTRYPOINT ["flower-superexec"]