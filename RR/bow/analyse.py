import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('prediction.csv',sep='\t')

y_true  =   df['annotation_label']
y_pred  =   df['predicted_label']


classification_report = classification_report(y_true,y_pred)
print(classification_report)

confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])

print(confusion_matrix)

sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap="OrRd")

plt.ylabel('Valeurs réelles')
plt.xlabel('Valeurs prédites')
plt.show()
