#from plotting.plot_methods.confusion_matrix_plot import confusion_matrix_plot
#from plotting.plot_methods.precision_recall_plot import precision_recall_plot
#from plotting.plot_methods.loss_accuracy_plot import loss_over_time_plot
from plotting.plot_methods.pr_auc_method import pr_auc_plot
from plotting.plot_methods.f1_plot_method import f1_plot



def plot_diagrams(pr_history, f1_history):
    #Udvid denne
    
    #confusion_matrix_plot(targets, preds)
    #loss_over_time_plot(train_losses, train_accuracies)

    pr_auc_plot(pr_history)
    f1_plot(f1_history)
