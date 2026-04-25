import matplotlib.pyplot as plt

def f1_plot(f1_history):
    """
    Plot F1 scores across rounds.
    """
    
    if not f1_history:
        print("No F1 history to plot")
        return
    
    # Extract rounds and F1 values
    rounds = [item[0] for item in f1_history]
    f1_values = [item[1] for item in f1_history]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, f1_values, marker='o', linewidth=2, markersize=8, label='F1 Score', color='blue')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score Across Rounds', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 1])  # F1 scores range from 0 to 1
    plt.tight_layout()
    
    plt.savefig("fl-backend/plotting/saved_plots/f1_plot.png", dpi=300)
    plt.close()