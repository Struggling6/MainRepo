#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict


@dataclass
class CheckResult:
    name: str
    status: str  # OK / WARNING / FAIL / SKIPPED
    message: str = ""
    details: Optional[str] = None


RESULTS: list[CheckResult] = []


def add(name: str, status: str, message: str = "", details: str | None = None) -> None:
    RESULTS.append(CheckResult(name, status, message, details))


def ok(name: str, message: str = "") -> None:
    add(name, "OK", message)


def warn(name: str, message: str = "", details: str | None = None) -> None:
    add(name, "WARNING", message, details)


def fail(name: str, message: str = "", details: str | None = None) -> None:
    add(name, "FAIL", message, details)


def skipped(name: str, message: str = "") -> None:
    add(name, "SKIPPED", message)


def fmt_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def try_import(module_name: str) -> Any | None:
    try:
        mod = importlib.import_module(module_name)
        ok(f"import:{module_name}", "Imported successfully")
        return mod
    except Exception as exc:
        fail(f"import:{module_name}", fmt_exc(exc), traceback.format_exc())
        return None


def has_init_py(folder: Path) -> bool:
    return (folder / "__init__.py").exists()


def summarize(strict: bool) -> int:
    order = {"FAIL": 0, "WARNING": 1, "OK": 2, "SKIPPED": 3}
    print("\n" + "=" * 88)
    print("FL BACKEND PROJECT VERIFICATION REPORT")
    print("=" * 88)
    for result in sorted(RESULTS, key=lambda r: (order.get(r.status, 9), r.name)):
        print(f"[{result.status:<7}] {result.name}")
        if result.message:
            print(f"          {result.message}")
        if result.details:
            lines = result.details.strip().splitlines()
            for line in lines[:10]:
                print(f"          {line}")
            if len(lines) > 10:
                print("          ...")

    total = len(RESULTS)
    ok_count = sum(r.status == "OK" for r in RESULTS)
    warn_count = sum(r.status == "WARNING" for r in RESULTS)
    fail_count = sum(r.status == "FAIL" for r in RESULTS)
    skipped_count = sum(r.status == "SKIPPED" for r in RESULTS)

    print("\n" + "-" * 88)
    print(
        f"Summary: total={total}, ok={ok_count}, warnings={warn_count}, failures={fail_count}, skipped={skipped_count}"
    )
    if fail_count == 0 and warn_count == 0:
        print("Overall result: HEALTHY")
    elif fail_count == 0:
        print("Overall result: MOSTLY HEALTHY (warnings present)")
    else:
        print("Overall result: FAILING")

    if strict:
        return 1 if (fail_count > 0 or warn_count > 0) else 0
    return 1 if fail_count > 0 else 0


def check_repo_shape(root: Path) -> None:
    expected = [
        "config.py",
        "client_app.py",
        "server_app.py",
        "FedProxWithSave.py",
        "data/registry.py",
        "data/lead_csv.py",
        "data/powergrid_csv.py",
        "training/train.py",
        "training/evaluate.py",
        "models/mlp.py",
        "models/LSTM.py",
        "models/supervised_cnn_transformer.py",
    ]
    for rel in expected:
        p = root / rel
        if p.exists():
            ok(f"path:{rel}", "Found")
        else:
            fail(f"path:{rel}", "Missing")

    for pkg in ["data", "training", "models"]:
        pkg_path = root / pkg
        if not pkg_path.exists():
            fail(f"package:{pkg}", "Folder missing")
            continue
        if has_init_py(pkg_path):
            ok(f"package:{pkg}", "Has __init__.py")
        else:
            if pkg == "models":
                fail(
                    f"package:{pkg}",
                    "Missing __init__.py",
                    "config.py imports from 'models', but models/ has no __init__.py exporting SupervisedTransformerCNN, LSTMModel, and MLPModel.",
                )
            else:
                warn(f"package:{pkg}", "No __init__.py present")


def check_models_exports(root: Path) -> None:
    models_init = root / "models" / "__init__.py"
    if not models_init.exists():
        fail(
            "models:exports",
            "Cannot verify exported model names because models/__init__.py is missing",
            "Expected something like:\nfrom .supervised_cnn_transformer import SupervisedTransformerCNN\nfrom .LSTM import LSTMModel\nfrom .mlp import MLPModel",
        )
        return

    text = models_init.read_text(encoding="utf-8", errors="ignore")
    missing = [
        name
        for name in ["SupervisedTransformerCNN", "LSTMModel", "MLPModel"]
        if name not in text
    ]
    if missing:
        fail("models:exports", f"models/__init__.py exists but seems to be missing exports: {missing}")
    else:
        ok("models:exports", "Expected model classes appear to be exported")


def check_config_build_path(config_mod: Any) -> None:
    cfg = getattr(config_mod, "CONFIG", None)
    if cfg is None:
        fail("config:CONFIG", "config.py imported but CONFIG was not found")
        return
    ok("config:CONFIG", f"Loaded {type(cfg).__name__}")

    for attr in ["task", "model", "data", "training", "federation", "evaluation"]:
        if hasattr(cfg, attr):
            ok(f"config:{attr}", f"Found {attr}")
        else:
            fail(f"config:{attr}", f"Missing CONFIG.{attr}")

    if hasattr(cfg, "model") and hasattr(cfg.model, "build"):
        ok("config:model.build", "Model config exposes build(input_dim=...)")
    else:
        fail("config:model.build", "CONFIG.model.build(...) not found")


def resolve_dataset_handler_from_config(data_registry_mod: Any, config_mod: Any):
    cfg = config_mod.CONFIG
    create_fn = getattr(data_registry_mod, "create_dataset_handler", None)
    if not callable(create_fn):
        raise RuntimeError("data.registry does not expose create_dataset_handler(config.data)")
    handler = create_fn(cfg.data)
    return cfg, handler


def instantiate_model_from_config(cfg: Any, metadata: dict[str, Any]):
    if "input_dim" not in metadata:
        raise RuntimeError("Dataset metadata missing required key 'input_dim'")
    return cfg.model.build(input_dim=metadata["input_dim"])


def first_batch(loader: Any):
    return next(iter(loader))


def build_balanced_smoke_loader(trainloader: Any, max_examples: int):
    import torch

    xs_by_class = defaultdict(list)
    ys_by_class = defaultdict(list)

    for batch in trainloader:
        x_batch, y_batch = batch

        x_batch = x_batch.cpu()
        y_batch = y_batch.cpu()

        for i in range(len(y_batch)):
            label = int(y_batch[i].item())
            xs_by_class[label].append(x_batch[i])
            ys_by_class[label].append(y_batch[i])

            num_classes_found = len([k for k, v in xs_by_class.items() if len(v) > 0])
            total_collected = sum(len(v) for v in xs_by_class.values())

            if num_classes_found >= 2 and total_collected >= max_examples:
                break

        num_classes_found = len([k for k, v in xs_by_class.items() if len(v) > 0])
        total_collected = sum(len(v) for v in xs_by_class.values())
        if num_classes_found >= 2 and total_collected >= max_examples:
            break

    available_classes = sorted(k for k, v in xs_by_class.items() if len(v) > 0)
    if len(available_classes) < 2:
        counts = {k: len(v) for k, v in xs_by_class.items()}
        raise ValueError(
            f"Could not build balanced smoke subset. Classes found: {available_classes}, counts: {counts}"
        )

    selected_x = []
    selected_y = []

    for cls in available_classes:
        selected_x.append(xs_by_class[cls].pop(0))
        selected_y.append(ys_by_class[cls].pop(0))

    while len(selected_x) < max_examples:
        added_any = False
        for cls in available_classes:
            if xs_by_class[cls]:
                selected_x.append(xs_by_class[cls].pop(0))
                selected_y.append(ys_by_class[cls].pop(0))
                added_any = True
                if len(selected_x) >= max_examples:
                    break
        if not added_any:
            break

    X = torch.stack(selected_x)
    y = torch.stack(selected_y)

    batch_size = min(len(X), max_examples)

    tiny_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
    )

    class_counts = {int(cls): int((y == cls).sum().item()) for cls in torch.unique(y)}
    return tiny_loader, class_counts


def check_flwr_wiring() -> None:
    flwr_mod = try_import("flwr")
    client_app_mod = try_import("client_app")
    server_app_mod = try_import("server_app")

    if flwr_mod is None:
        warn(
            "flwr:wiring",
            "Skipping Flower wiring checks because 'flwr' could not be imported",
        )
        return

    if client_app_mod is None or server_app_mod is None:
        warn(
            "flwr:wiring",
            "Skipping Flower app checks because client_app or server_app failed to import",
        )
        return

    client_app_obj = getattr(client_app_mod, "app", None)
    server_app_obj = getattr(server_app_mod, "app", None)

    if client_app_obj is None:
        fail("flwr:client_app.app", "client_app imported but no 'app' object was found")
    else:
        ok("flwr:client_app.app", f"Found app object of type {type(client_app_obj).__name__}")

    if server_app_obj is None:
        fail("flwr:server_app.app", "server_app imported but no 'app' object was found")
    else:
        ok("flwr:server_app.app", f"Found app object of type {type(server_app_obj).__name__}")

    try:
        from flwr.client import ClientApp
        if client_app_obj is not None:
            if isinstance(client_app_obj, ClientApp):
                ok("flwr:client_app.type", "client_app.app is a Flower ClientApp")
            else:
                warn(
                    "flwr:client_app.type",
                    f"client_app.app exists but is not a ClientApp (got {type(client_app_obj).__name__})",
                )
    except Exception as exc:
        warn("flwr:client_app.type", f"Could not validate ClientApp type: {fmt_exc(exc)}")

    try:
        from flwr.server import ServerApp
        if server_app_obj is not None:
            if isinstance(server_app_obj, ServerApp):
                ok("flwr:server_app.type", "server_app.app is a Flower ServerApp")
            else:
                warn(
                    "flwr:server_app.type",
                    f"server_app.app exists but is not a ServerApp (got {type(server_app_obj).__name__})",
                )
    except Exception as exc:
        warn("flwr:server_app.type", f"Could not validate ServerApp type: {fmt_exc(exc)}")


def check_dataset_and_model_flow(root: Path, config_mod: Any, data_registry_mod: Any, train_mod: Any, args: argparse.Namespace) -> None:
    try:
        cfg, handler = resolve_dataset_handler_from_config(data_registry_mod, config_mod)
        ok("dataset:init", f"Created handler {type(handler).__name__} for {cfg.data.name}")
    except Exception as exc:
        fail("dataset:init", fmt_exc(exc), traceback.format_exc())
        return

    try:
        metadata = handler.get_metadata()
        if not isinstance(metadata, dict):
            fail("dataset:metadata", f"Expected dict, got {type(metadata).__name__}")
            return
        required = ["input_dim", "num_classes", "num_samples", "task_type", "data_format"]
        missing = [k for k in required if k not in metadata]
        if missing:
            warn("dataset:metadata", f"Missing metadata keys: {missing}", str(metadata))
        else:
            ok("dataset:metadata", str(metadata))
    except Exception as exc:
        fail("dataset:metadata", fmt_exc(exc), traceback.format_exc())
        return

    partition_id = args.partition_id
    try:
        trainloader, testloader = handler.get_dataloaders(partition_id=partition_id)
        ok("dataset:dataloaders", f"Obtained train/test loaders for partition_id={partition_id}")
    except Exception as exc:
        fail("dataset:dataloaders", fmt_exc(exc), traceback.format_exc())
        return

    try:
        batch = first_batch(trainloader)
        x, y = batch
        x_shape = tuple(x.shape) if hasattr(x, "shape") else type(x).__name__
        y_shape = tuple(y.shape) if hasattr(y, "shape") else type(y).__name__
        ok("dataset:first_batch", f"x.shape={x_shape}, y.shape={y_shape}")

        if hasattr(x, "ndim") and x.ndim not in (2, 3):
            warn("dataset:input_rank", f"Expected rank 2 or 3 input for current models, got rank {x.ndim}")
    except Exception as exc:
        fail("dataset:first_batch", fmt_exc(exc), traceback.format_exc())
        return

    try:
        import torch
        if torch.isnan(x.float()).any():
            warn("dataset:nan_input", "First batch contains NaNs")
        if torch.isinf(x.float()).any():
            warn("dataset:inf_input", "First batch contains inf values")
    except Exception:
        pass

    try:
        model = instantiate_model_from_config(cfg, metadata)
        ok("model:init", f"Built model {type(model).__name__}")
    except Exception as exc:
        fail("model:init", fmt_exc(exc), traceback.format_exc())
        return

    try:
        import torch
        model.eval()
        with torch.no_grad():
            out = model(x)
        out_shape = tuple(out.shape) if hasattr(out, "shape") else type(out).__name__
        ok("model:forward", f"Forward pass succeeded with output shape {out_shape}")

        if hasattr(out, "shape") and hasattr(y, "shape") and len(out.shape) > 0 and len(y.shape) > 0:
            if out.shape[0] != y.shape[0]:
                warn("model:batch_mismatch", f"Output batch {out.shape[0]} != target batch {y.shape[0]}")
    except Exception as exc:
        fail("model:forward", fmt_exc(exc), traceback.format_exc())
        return

    if args.skip_train:
        skipped("train:smoke", "Skipped by user request")
        return

    try:
        import torch
        train_model = getattr(train_mod, "train_model", None)
        if not callable(train_model):
            fail("train:smoke", "training.train does not expose train_model(...)")
            return

        try:
            tiny_loader, class_counts = build_balanced_smoke_loader(
                trainloader=trainloader,
                max_examples=args.max_train_batch,
            )
            ok("train:smoke_subset", f"Built balanced smoke subset with class counts {class_counts}")
        except Exception as subset_exc:
            warn(
                "train:smoke_subset",
                f"Skipping training smoke test: {subset_exc}",
            )
            skipped("train:smoke", "No usable balanced subset could be built")
            return

        import copy
        smoke_training_config = copy.deepcopy(cfg.training)
        if hasattr(smoke_training_config, "epochs"):
            smoke_training_config.epochs = min(smoke_training_config.epochs, 2)
        if hasattr(smoke_training_config, "early_stopping_patience"):
            smoke_training_config.early_stopping_patience = 1

        result = train_model(
            model=model,
            trainloader=tiny_loader,
            training_config=smoke_training_config,
            model_config=cfg.model,
            device=torch.device("cpu"),
        )
        ok("train:smoke", f"train_model succeeded: {result}")
    except Exception as exc:
        fail("train:smoke", fmt_exc(exc), traceback.format_exc())


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify fl_backend project wiring")
    parser.add_argument("--project-root", default=".", help="Path to repo root")
    parser.add_argument("--partition-id", type=int, default=0, help="Partition id to test")
    parser.add_argument("--skip-train", dest="skip_train", action="store_true", help="Skip calling training.train_model")
    parser.add_argument("--strict", action="store_true", help="Warnings also cause non-zero exit")
    parser.add_argument("--max-train-batch", type=int, default=16, help="Max examples used for the smoke training step")
    parser.add_argument("--verify-flwr", action="store_true", help="Verify Flower imports and app wiring")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    if not root.exists():
        print(f"Project root does not exist: {root}")
        return 2

    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ok("cwd", str(root))
    check_repo_shape(root)
    check_models_exports(root)

    try:
        import torch
        ok("import:torch", f"PyTorch {torch.__version__}")
    except Exception as exc:
        fail("import:torch", fmt_exc(exc), traceback.format_exc())

    try:
        import pandas  # noqa: F401
        ok("import:pandas", "Imported successfully")
    except Exception as exc:
        fail("import:pandas", fmt_exc(exc), traceback.format_exc())

    try:
        import numpy  # noqa: F401
        ok("import:numpy", "Imported successfully")
    except Exception as exc:
        fail("import:numpy", fmt_exc(exc), traceback.format_exc())

    config_mod = try_import("config")
    if config_mod is None:
        warn(
            "hint:config_import",
            "Core verification stopped early because config.py could not be imported",
            "Fixing the config/models import path should unlock the deeper dataset/model/training checks.",
        )
        return summarize(args.strict)

    check_config_build_path(config_mod)

    data_registry_mod = try_import("data.registry")
    train_mod = try_import("training.train")
    try_import("training.evaluate")
    try_import("client_app")
    try_import("server_app")
    try_import("FedProxWithSave")

    if args.verify_flwr:
        check_flwr_wiring()

    if data_registry_mod is None or train_mod is None:
        warn(
            "hint:deeper_checks",
            "Stopping before end-to-end checks because a required module import failed",
        )
        return summarize(args.strict)

    check_dataset_and_model_flow(root, config_mod, data_registry_mod, train_mod, args)
    return summarize(args.strict)


if __name__ == "__main__":
    raise SystemExit(main())