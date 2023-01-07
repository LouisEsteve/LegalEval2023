import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# OUVERTURE DU FICHIER
df = pd.read_csv('prediction.csv',sep='\t')

y_true  =   df['annotation_label']
y_pred  =   df['predicted_label']


# RAPPORT DE CLASSIFICATION
classification_report = classification_report(y_true,y_pred)
print(classification_report)


# MATRICE DE CONFUSION
confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)


# AFFICHER LA MATRCIE DE CONFUSION PNG
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap="OrRd")
plt.ylabel('Valeurs réelles')
plt.xlabel('Valeurs prédites')
#plt.show()

# CALCULER l'OVERLAP
overlap = pd.crosstab(df['annotation_label'], df['predicted_label']).apply(lambda x: x[x.index != x.name].sum(), axis=1)
overlap = overlap.sort_values(ascending=False)
print(overlap)
