import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def confusion_matrix_plot(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
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

    plt.savefig("fl-backend/plotting/saved_plots/confusion_matrix.png")
    plt.close()

    precision = cm[1][1] / (cm[1][1] + cm[0][1]) #Når modellen siger 1, hvor ofte er det rigtigt 
    #TP / (TP + FP)
    print("precision : ", precision)
    recall = cm[1][1] / (cm[1][1] + cm[1][0]) #Af alle de rigtige 1’ere, hvor mange fandt modellen    
    #TP / (TP + FN)
    print("recall : ", recall)

    accuracy = (cm[1][1] + cm[0][0]) / (cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) #hvor mange preds er rigtige i alt
    #(TP + TN) / (TP + TN + FP + FN)
    print(accuracy)


    '''
    - Kombination af precision OG recall, fordi hver for sig ser du kun de seperate eksempleer
    - Men med kombinationen af begge, ser vi en mere præcis measurement af hvor mange predictions vi faktisk får korrekt.
    - Så jo højere den er, jo bedere er modellen er til både at finde positives og undgå fejl
    '''
    f1_score = 2 * (precision*recall)/(precision + recall)

    print("F1 score : ", f1_score)

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
