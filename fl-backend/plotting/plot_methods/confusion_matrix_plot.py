import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path

def confusion_matrix_plot(cm, save_dir: Path):

    '''
    Cm ville returnere : 
    [[TN, FP],
    [FN, TP]]

    TN = True negative, så true = 0 + Pred = 0, the model correctly predicted negative 
    FP = False positive, True = 0 + Pred = 1, the model icorrectly predicted positive
    FN = False negative, True = 1 + Pred = 0, the model icorrectly predicted negative
    TP = True positive, True = 1 + Pred = 1, the model correctly predicted positive

    '''

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d",  xticklabels=["True 0", "Pred 1"], yticklabels=["Pred 0", "True 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix plot to: {output_path}")

'''
How to interpret : 
    Correctness : 
        - TN og TP
        - Hvis disse er høje, betyder det at modellen predicter korrekt det meste at tiden
    
    Errors : 
        - Dette er FP og FN
        - Dette betyder at modellen laver false alarms og misser positives
        - Altså det er ikke godt hvis disse 2 er høje
    
    Specifikt for anomaly detection : 
        - FN er et stort problem
        - Det betyder modellen tror patterns som skaber anomalies, er normale og classifier dem som normal

    Så med vores MLP model, får vi : 
         cm = [[ 24 176]
              [  7 786]]

        precision :  0.825668449197861
        recall :  0.9735182849936949
        0.8147029204431017
        F1 score :  0.8935185185185186

    Dette betyder : 
        - Vores model finder næsten ALLE positives fordi recall er høj
        - Vores model er meget biased mod klasse 1, altså at ting er en anomaly fordi vi kan se at FP og TP dominerer matricen
        - Men accuracy er høj, hvilket i dette tilfælde er misvisende
        - Vi kan SE at modellen er meget ubalanceret baseret på matricen
        - så derfor kan vi antage at det er datasettet der har for mange positives
        - så den er alt for god til at finde positives, men ikke gode til at finde negatives. 
        - MEN siden vores model er til anomaly detection, er dette ikke nødvendigvis en dårlig ting.
        - Ofte mener folk at, i dette scenarie, at false positives er bedere end false negatives
        - Så hvis FN var meget høj, ville det være et STORT problem for os
'''
