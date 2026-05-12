#from plotting.plot_methods.confusion_matrix_plot import confusion_matrix_plot
#from plotting.plot_methods.precision_recall_plot import precision_recall_plot
#from plotting.plot_methods.loss_accuracy_plot import loss_over_time_plot
from pathlib import Path

import os

import matplotlib

matplotlib.use("Agg")

from plotting.plot_methods.pr_auc_method import pr_auc_plot
from plotting.plot_methods.f1_plot_method import f1_plot
from plotting.plot_methods.roc_auc_method import roc_auc_plot

BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = Path(os.environ.get("PLOTS_DIR", str(BASE_DIR / "saved_plots")))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def plot_diagrams(pr_history, f1_history, roc_history):
    #Udvid denne
    
    #confusion_matrix_plot(targets, preds)
    #loss_over_time_plot(train_losses, train_accuracies)

    pr_auc_plot(pr_history, save_dir=SAVE_DIR)
    f1_plot(f1_history, save_dir=SAVE_DIR)
    roc_auc_plot(roc_history, save_dir=SAVE_DIR)
