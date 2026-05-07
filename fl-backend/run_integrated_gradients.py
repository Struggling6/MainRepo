from __future__ import annotations

import argparse

from local_experiment import CONFIG
from interpretability.integrated_gradients import run_integrated_gradients
from interpretability.visualization import (
    plot_feature_importance,
    plot_time_importance,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Integrated Gradients on a saved federated model."
    )

    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Number of test batches to explain. Use -1 for all batches.",
    )
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    max_batches = None if args.max_batches < 0 else args.max_batches

    npz_path = run_integrated_gradients(
        config=CONFIG,
        partition_id=args.partition_id,
        max_batches=max_batches,
        target_class=args.target_class,
    )

    if args.plot:
        plot_feature_importance(npz_path)

        try:
            plot_time_importance(npz_path)
        except ValueError as exc:
            print(exc)


if __name__ == "__main__":
    main()