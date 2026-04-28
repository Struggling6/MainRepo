#!/usr/bin/env python3
"""
Detects GPU type (CPU / CUDA / ROCm), then generates:
  - fl-backend/requirements.txt   with the correct torch index URL
  - compose.yml                   with the correct device mappings

Usage:
    python preflight.py
    python preflight.py --variant rocm
    python preflight.py --num-clients 3 --cpus 2.0 --mem-limit 8g
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

REQUIREMENTS_BASE = [
    "flwr[simulation]>=1.27.0",
    "pandas==3.0.2",
    "numpy==2.4.4",
    "scikit-learn==1.8.0",
    "matplotlib==3.10.9",
    "transformers==5.6.2",
    "captum==0.9.0",
]

TORCH_PINS = {
    "cpu": "torch==2.11.0",
    "cuda": "torch==2.11.0",
    "rocm": "torch",
}

BASE_INDEX_URL = "https://pypi.org/simple"

TORCH_INDEX_URLS = {
    "cuda": "https://download.pytorch.org/whl/cu128",
    "rocm": "https://download.pytorch.org/whl/rocm7.2",
}

RESOURCE_FIELDS = ["cpus", "mem_limit"]

REQUIREMENTS_OUTPUT = Path("fl-backend/requirements.txt")
COMPOSE_OUTPUT      = Path("compose.yml")
BUILD_CONTEXT       = "./fl-backend"
DOCKERFILE          = "Dockerfile"
START_PORT          = 9094
DOCKERIGNORE_PATH   = Path("fl-backend/.dockerignore")
DOCKERIGNORE_DATASET_ENTRY = "datasets"

# ------------------------------------------------------------------ #
#  GPU detection                                                       #
# ------------------------------------------------------------------ #

def detect_variant() -> str:
    """Return 'cuda', 'rocm', or 'cpu'."""
    if shutil.which("rocm-smi") or Path("/dev/kfd").exists():
        try:
            r = subprocess.run(["rocm-smi", "--showproductname"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                print("Detected: ROCm GPU")
                return "rocm"
        except Exception:
            pass
        print("Detected: ROCm GPU (via /dev/kfd)")
        return "rocm"

    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                print(f"Detected: CUDA GPU ({r.stdout.strip().splitlines()[0]})")
                return "cuda"
        except Exception:
            pass

    print("Detected: CPU only")
    return "cpu"

# ------------------------------------------------------------------ #
#  requirements.txt                                                    #
# ------------------------------------------------------------------ #

def write_requirements(variant: str) -> None:
    lines = [f"--index-url {BASE_INDEX_URL}"]
    if variant in TORCH_INDEX_URLS:
        lines.append(f"--extra-index-url {TORCH_INDEX_URLS[variant]}")
    lines += [TORCH_PINS.get(variant, "torch")]
    lines += REQUIREMENTS_BASE
    content = "\n".join(lines) + "\n"
    REQUIREMENTS_OUTPUT.write_text(content)
    print(f"Wrote {REQUIREMENTS_OUTPUT} [{variant}]\n{'-' * 40}\n{content}")


def ensure_dockerignore_datasets() -> None:
    if not DOCKERIGNORE_PATH.exists():
        DOCKERIGNORE_PATH.write_text(f"{DOCKERIGNORE_DATASET_ENTRY}\n", encoding="utf-8")
        print(f"Created {DOCKERIGNORE_PATH} and added {DOCKERIGNORE_DATASET_ENTRY}.")
        return

    content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    if DOCKERIGNORE_DATASET_ENTRY in {line.strip() for line in content.splitlines()}:
        return

    updated = content
    if updated and not updated.endswith("\n"):
        updated += "\n"
    updated += f"{DOCKERIGNORE_DATASET_ENTRY}\n"
    DOCKERIGNORE_PATH.write_text(updated, encoding="utf-8")
    print(f"Added {DOCKERIGNORE_DATASET_ENTRY} to {DOCKERIGNORE_PATH}.")

# ------------------------------------------------------------------ #
#  Compose helpers                                                     #
# ------------------------------------------------------------------ #

def _yaml_list(key: str, items: list[str], indent: int = 4) -> str:
    pad = " " * indent
    return f"{pad}{key}:\n" + "".join(f"{pad}  - {item}\n" for item in items)


def _gpu_block(variant: str) -> str:
    if variant == "rocm":
        return (
            _yaml_list("devices", ["/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"])
            + _yaml_list("group_add", ["render", "video"])
        )
    if variant == "cuda":
        return (
            "    deploy:\n"
            "      resources:\n"
            "        reservations:\n"
            "          devices:\n"
            "            - driver: nvidia\n"
            "              count: all\n"
            "              capabilities: [gpu]\n"
        )
    return ""


def _resource_block(res: dict[str, Any]) -> str:
    lines: list[str] = []

    for field in RESOURCE_FIELDS:
        if (val := res.get(field)) is not None:
            lines.append(f'cpus: "{val}"' if field == "cpus" else f"{field}: {val}")

    return ("".join(f"    {l}\n" for l in lines)) if lines else ""

# ------------------------------------------------------------------ #
#  Compose sections                                                    #
# ------------------------------------------------------------------ #

def _header() -> str:
    return f"""\
name: fl-backend

x-superexec-image: &superexec_image fl-backend-superexec:local

services:
  superlink:
    image: flwr/superlink:1.27.0
    command:
      - --insecure
      - --serverappio-api-address
      - 0.0.0.0:9091
      - --fleet-api-address
      - 0.0.0.0:9092
      - --fleet-api-type
      - grpc-rere
    ports:
      - "9093:9093"
      - "9091:9091"
      - "9092:9092"

  superexec-serverapp:
        image: *superexec_image
    build:
      context: {BUILD_CONTEXT}
      dockerfile: {DOCKERFILE}
    command:
      - --insecure
      - --plugin-type
      - serverapp
      - --appio-api-address
      - superlink:9091
    depends_on:
      - superlink
    stop_signal: SIGINT
    volumes:
      - ./fl-backend/checkpoints:/app/checkpoints

"""


def _supernode(n: int, partition_id: int, port: int, num_clients: int) -> str:
    return f"""\
  supernode-{n}:
    image: flwr/supernode:1.27.0
    command:
      - --insecure
      - --superlink
      - superlink:9092
      - --clientappio-api-address
      - 0.0.0.0:{port}
      - --isolation
      - process
      - --node-config
      - "partition-id={partition_id} num-partitions={num_clients}"
    depends_on:
      - superlink

"""


def _clientapp(n: int, port: int, variant: str, res: dict[str, Any]) -> str:
    return (
        f"  superexec-clientapp-{n}:\n"
        f"    image: *superexec_image\n"
        f"    build:\n"
        f"      context: {BUILD_CONTEXT}\n"
        f"      dockerfile: {DOCKERFILE}\n"
        f"    command:\n"
        f"      - --insecure\n"
        f"      - --plugin-type\n"
        f"      - clientapp\n"
        f"      - --appio-api-address\n"
        f"      - supernode-{n}:{port}\n"
        f"    depends_on:\n"
        f"      - supernode-{n}\n"
        f"    stop_signal: SIGINT\n"
        f"    volumes:\n"
        f"      - ./fl-backend/datasets:/app/datasets\n"
        f"{_gpu_block(variant)}"
        f"{_resource_block(res)}"
        f"\n"
    )

# ------------------------------------------------------------------ #
#  compose.yml                                                         #
# ------------------------------------------------------------------ #

def write_compose(variant: str, args: argparse.Namespace) -> None:
    res: dict[str, Any] = {f: getattr(args, f, None) for f in RESOURCE_FIELDS}

    parts = [_header()]
    for i in range(args.num_clients):
        n, port = i + 1, START_PORT + i
        parts += [_supernode(n, i, port, args.num_clients),
                  _clientapp(n, port, variant, res)]

    COMPOSE_OUTPUT.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {COMPOSE_OUTPUT} with {args.num_clients} client(s) [{variant}].")

# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate requirements.txt and compose.yml for the detected GPU variant."
    )
    p.add_argument("--variant", choices=["cpu", "cuda", "rocm"], default=None,
                   help="Force a GPU variant instead of auto-detecting")
    p.add_argument("--num-clients", type=int, default=2,
                   help="Number of federated clients (default: 2)")
    p.add_argument("--cpus", default=None,
                   help='CPU limit per clientapp, e.g. "2.0"')
    p.add_argument("--mem-limit", dest="mem_limit", default=None,
                   help='Hard memory limit per clientapp, e.g. "8g"')
    p.add_argument("--mem-reservation", dest="mem_reservation", default=None,
                   help='Soft memory limit per clientapp, e.g. "6g"')
    args = p.parse_args()

    if args.num_clients < 1:
        raise ValueError("--num-clients must be at least 1")

    variant = args.variant or detect_variant()
    ensure_dockerignore_datasets()
    write_requirements(variant)
    write_compose(variant, args)


if __name__ == "__main__":
    main()