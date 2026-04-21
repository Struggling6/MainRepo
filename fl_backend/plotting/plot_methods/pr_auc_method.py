import matplotlib.pyplot as plt
import numpy as np

def pr_auc_plot(pr_history):
    """
    Plots PR_AUC saved values from each round.
    The closer the curve is to the top left corner, the better the model is at preciting correctly while not making false positives. 
    """
    
    if not pr_history:
        print("No PR-AUC history to plot")
        return
    
    # Extract rounds and PR-AUC values
    rounds = [item[0] for item in pr_history]
    pr_auc_values = [item[1] for item in pr_history]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, pr_auc_values, marker='o', linewidth=2, markersize=8, label='PR-AUC')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('PR-AUC', fontsize=12)
    plt.title('PR-AUC Across Rounds', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save and show
    plt.savefig("fl_backend/plotting/saved_plots/pr_auc_plot.png", dpi=300)
    plt.close()


# Example usage:
# pr_history = [(1, 0.0063107944772530155), (2, 0.0063107944772530155)]
# pr_auc_plot(pr_history)