import matplotlib.pyplot as plt


def loss_over_time_plot(train_losses, train_accuracies):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(train_accuracies, label="Train Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("fl_backend/plotting/saved_plots/loss_curve.png", dpi=300)
    plt.close()

'''
How to interpret : 
    - Accuracy = Hvor mange predictions er korrekte
        - Så høj accuraccy betyder mange af predictions er korrekte 
    - Loss = Hvor forkerte predictionerne er
        - Så lav loss betyder at modellen bliver mere sikker på dens korrekte predictions 

    Så hvis : 
    - Accuracy er høj og loss er lav : 
        - Så er modellen god, fordi de følger hindanden
    
    - Accuracy er lav og loss er høj : 
        - Så er det under fitting og betyder at modellen ikke lærer nok
        - Så skal learning rate justeres 
    
    BONUS!!!
    - Train accuracy er høj, train loss er lav, validation accuracy er lav og validation loss er høj
        - Så betyder det overfitting og modellen lærer alt for specifikt til det brugte dataset 
        - Så skal man bruge dropout eller weight decay så den ikke vægter for meget på en specific neuron
        - Lige nu plotter vi ikke validation stats så dette vises ikke i dette diagram 
'''