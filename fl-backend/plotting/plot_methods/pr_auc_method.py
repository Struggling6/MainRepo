from pathlib import Path

import matplotlib.pyplot as plt


def pr_auc_plot(pr_history, save_dir: Path):
    """
    Plots PR-AUC values from each round.
    """

    if not pr_history:
        print("No PR-AUC history to plot")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    rounds = [item[0] for item in pr_history]
    pr_auc_values = [item[1] for item in pr_history]

    plt.figure(figsize=(10, 6))
    plt.plot(
        rounds,
        pr_auc_values,
        marker="o",
        linewidth=2,
        markersize=8,
        label="PR-AUC",
    )

    plt.xlabel("Round", fontsize=12)
    plt.ylabel("PR-AUC", fontsize=12)
    plt.title("PR-AUC Across Rounds", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = save_dir / "pr_auc_plot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved PR-AUC plot to: {output_path}")