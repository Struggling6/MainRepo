from pathlib import Path

import matplotlib.pyplot as plt


def f1_plot(f1_history, save_dir: Path):
    """
    Plot F1 scores across rounds.
    """

    if not f1_history:
        print("No F1 history to plot")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    rounds = [item[0] for item in f1_history]
    f1_values = [item[1] for item in f1_history]

    plt.figure(figsize=(10, 6))
    plt.plot(
        rounds,
        f1_values,
        marker="o",
        linewidth=2,
        markersize=8,
        label="F1 Score",
    )

    plt.xlabel("Round", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("F1 Score Across Rounds", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 1])
    plt.tight_layout()

    output_path = save_dir / "f1_plot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved F1 plot to: {output_path}")