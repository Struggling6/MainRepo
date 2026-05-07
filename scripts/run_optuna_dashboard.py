#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_storage() -> str:
    base_dir = _repo_root() / "fl-backend"
    return f"sqlite:////{str(base_dir).lstrip('/')}/optuna_study.db"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Optuna Dashboard")
    parser.add_argument(
        "--storage",
        type=str,
        default=os.environ.get("STORAGE", _default_storage()),
        help="Optuna storage URL (default: fl-backend/optuna_study.db)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8080")),
        help="Port to bind (default: 8080)",
    )
    args = parser.parse_args()

    try:
        import optuna_dashboard  # noqa: F401
    except ImportError:
        print("optuna-dashboard is not installed in the current environment.")
        print("Install deps with: python scripts/generate.py && uv pip install -r fl-backend/requirements.txt")
        return 1

    print("Starting Optuna Dashboard")
    print(f"  Storage : {args.storage}")
    print(f"  Host    : {args.host}")
    print(f"  Port    : {args.port}")

    cmd = [
        "optuna-dashboard",
        "--host",
        args.host,
        "--port",
        str(args.port),
        args.storage,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
