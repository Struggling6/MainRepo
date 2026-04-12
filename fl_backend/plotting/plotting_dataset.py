import matplotlib.pyplot as plt
import pandas as pd

def plot_single_row():
    """Plot a single row from the CSV dataset."""
    # Læs CSV filen
    df = pd.read_csv('/Users/sona/Desktop/Fbwg projekt/MainRepo/fl_backend/datasets/data1.csv')
    
    # Tag første række (index 0)
    row = df.iloc[0]
    
    # Fjern 'marker' kolonne hvis den findes (sidste kolonne)
    if 'marker' in row.index:
        row = row.drop('marker')
    
    # Plot
    plt.figure(figsize=(16, 5))
    plt.plot(row.values, marker='o', markersize=3, linewidth=0.8)
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.title('First Row of Dataset')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_single_row()