import os
from pathlib import Path
import pandas as pd

from tqdm.auto import tqdm

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# DATA
CUR_DIR = Path(__file__).parent
BASE_DIR = Path(CUR_DIR).parent
DATA_DIR = os.path.join(BASE_DIR, "Data")


# FONCTIONS
def evaluate_model(label, pred):
    """
    Créer, imprimer à la console et sauvegarder un rapport des scores des métriques d'évaluation
    """
    cr = metrics.classification_report(label, pred, zero_division=1, digits=3)
    print(cr, "\n")
    
    cr = metrics.classification_report(label, pred, zero_division=1, digits=3, output_dict=True)
    df_cr = pd.DataFrame(cr)
    filepath = os.path.join(CUR_DIR, "dev_classification_report.csv")
    df_cr.to_csv(filepath, sep=";", encoding="UTF-8")
    print(f"Métriques d'évaluation sauvegardés @ {filepath}")

print(CUR_DIR)
# CODE
file = os.path.join(CUR_DIR, "dev_pred.csv")

df = pd.read_csv(file, sep=";", encoding="UTF-8")

# Evaluer les prédictions
evaluate_model(df["label"], df["prediction"])