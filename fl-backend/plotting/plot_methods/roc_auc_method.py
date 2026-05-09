from pathlib import Path

import matplotlib.pyplot as plt


def roc_auc_plot(roc_history, save_dir: Path):
    """
    Plots ROC-AUC values from each round.
    """

    if not roc_history:
        print("No ROC-AUC history to plot")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    rounds = [item[0] for item in roc_history]
    roc_auc_values = [item[1] for item in roc_history]

    plt.figure(figsize=(10, 6))
    plt.plot(
        rounds,
        roc_auc_values,
        marker="o",
        linewidth=2,
        markersize=8,
        label="ROC-AUC",
    )

    plt.xlabel("Round", fontsize=12)
    plt.ylabel("ROC-AUC", fontsize=12)
    plt.title("ROC-AUC Across Rounds", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = save_dir / "roc_auc_plot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved ROC-AUC plot to: {output_path}")