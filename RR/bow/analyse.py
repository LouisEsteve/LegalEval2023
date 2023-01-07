import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(filename):
    df = pd.read_csv(filename, sep='\t')
    return df

def get_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print(report)

def get_confusion_matrix(y_true, y_pred):
    matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print(matrix)

def plot_confusion_matrix(y_true, y_pred):
    matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(matrix, annot=True, fmt='g', cmap="OrRd")
    plt.ylabel('Valeurs réelles')
    plt.xlabel('Valeurs prédites')
    plt.show()

def get_overlap(df, y_true_col, y_pred_col):
    overlap = pd.crosstab(df[y_true_col], df[y_pred_col]).apply(lambda x: x[x.index != x.name].sum(), axis=1)
    overlap = overlap.sort_values(ascending=False)
    print(overlap)

def main():
    df = load_data('prediction.csv')
    y_true = df['annotation_label']
    y_pred = df['predicted_label']

    get_classification_report(y_true, y_pred)
    get_confusion_matrix(y_true, y_pred)
    get_overlap(df, 'annotation_label', 'predicted_label')
    plot_confusion_matrix(y_true, y_pred)

if __name__ == '__main__':
    main()
