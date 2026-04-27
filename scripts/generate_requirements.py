#!/usr/bin/env python3
"""
generate_requirements.py

Detects GPU type (CPU / CUDA / ROCm) and writes a requirements.txt
with the appropriate torch index URL baked in.

Usage:
    python generate_requirements.py
    python generate_requirements.py --output requirements.txt
    python generate_requirements.py --variant rocm  # force a variant
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ------------------------------------------------------------------ #
#  Base dependencies (mirrors pyproject.toml)                         #
# ------------------------------------------------------------------ #
BASE_DEPS = [
    "flwr[simulation]>=1.27.0",
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "transformers",
]

TORCH_DEPS = ["torch"]

INDEX_URLS = {
    "cuda": "https://download.pytorch.org/whl/cu124",
    "rocm": "https://download.pytorch.org/whl/rocm6.3",
}


# ------------------------------------------------------------------ #
#  GPU detection                                                       #
# ------------------------------------------------------------------ #
def detect_variant() -> str:
    """Return 'cuda', 'rocm', or 'cpu'."""

    # --- ROCm: check for AMD GPU via rocm-smi or /dev/kfd --------------
    if shutil.which("rocm-smi") or Path("/dev/kfd").exists():
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                print("Detected: ROCm GPU")
                return "rocm"
        except Exception:
            pass
        # /dev/kfd present but rocm-smi missing — still likely ROCm
        print("Detected: ROCm GPU (via /dev/kfd)")
        return "rocm"

    # --- CUDA: check for nvidia-smi ------------------------------------
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"Detected: CUDA GPU ({result.stdout.strip().splitlines()[0]})")
                return "cuda"
        except Exception:
            pass

    print("Detected: CPU only")
    return "cpu"


# ------------------------------------------------------------------ #
#  requirements.txt generation                                         #
# ------------------------------------------------------------------ #
def build_requirements(variant: str) -> str:
    lines = []

    # torch + torchvision with index URL if needed
    if variant in INDEX_URLS:
        lines.append(f"--extra-index-url {INDEX_URLS[variant]}")

    lines.extend(TORCH_DEPS)
    lines.extend(BASE_DEPS)

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Generate requirements.txt for detected GPU.")
    parser.add_argument("--variant", choices=["cpu", "cuda", "rocm"], default=None,
                        help="Force a specific variant instead of auto-detecting")
    args = parser.parse_args()

    variant = args.variant or detect_variant()
    content = build_requirements(variant)

    output_path = Path("fl-backend/requirements.txt")
    output_path.write_text(content)

    print(f"\nWrote {output_path} for variant: {variant}")
    print("-" * 40)
    print(content)


if __name__ == "__main__":
    main()