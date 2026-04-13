from plotting.plot_methods.confusion_matrix_plot import confusion_matrix_plot
from plotting.plot_methods.precision_recall_plot import precision_recall_plot
from plotting.plot_methods.loss_accuracy_plot import loss_over_time_plot

def plot_diagrams(targets, preds, train_losses, train_accuracies):
    #Udvid denne når den nye config struktur er blevet implementeret så det hele passer
    confusion_matrix_plot(targets, preds)
    loss_over_time_plot(train_losses, train_accuracies)