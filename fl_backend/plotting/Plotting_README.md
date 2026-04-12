# Model Evaluation and Plotting Setup

Når der bliver tilføjet en ny plotting metode, skal man i denne fil skrive : 
- Hvilke input metoden har brug for
- Hvilken del af modellen skal generer værdierne og om nogle returns skal ændres 
- Hvordan den skal læses?


## Confusion Matrix

For at kunne beregne en confusion matrix skal evaluation-metoden returnere følgende:

- `y_true`
- `y_pred`

### Krav

- Begge tensors skal indeholde værdier for **hver batch**
- De skal repræsentere:
  - `y_true`: de rigtige labels
  - `y_pred`: modellens predictions

### Return format

Disse værdier skal returneres sammen med de andre metrics fra evaluation-metoden.

---

## Loss Over Time Plot

Training-metoden skal returnere:

- `train_losses`
- `train_accuracies`

### Krav

- Værdierne skal beregnes **for hver epoch**
- De skal gemmes i arrays eller lister

### Return format

Disse værdier skal returneres sammen med de øvrige træningsresultater.

### Fremtidig udvidelse

Det vil senere være relevant også at tilføje:

- `val_losses`
- `val_accuracies`

Dette gør det muligt at:

- analysere generalisering
- identificere overfitting

---

## Plotting Configuration

Plotting-systemet er endnu ikke fuldt udviklet, men tanken er :

### Struktur

- Specifikke plotting-metoder placeres i mappen:

```python
plot_methods/
