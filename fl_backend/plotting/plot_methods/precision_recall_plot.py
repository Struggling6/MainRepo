import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

'''
- Det her virker virker kun med probabilies, ikke 0/1 predictions
- Men det skal vise trade-off mellem precision og recall
- Det skal hjælpe dig med at adjust til threshhold til hvornår modellen predicter noget er en anomaly
- Så HVIS :
    - Recall er høj og precision er lav, skal threshholden være lavere
    - Recall er lav og precision er høj, skal thredhold være højere
    - Kurven skal så være meget høj, falde meget langsomt og dække et stort område
'''
def precision_recall_plot(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig("fl_backend/plotting/saved_plots/percision_recall_plot.png")
    plt.close()