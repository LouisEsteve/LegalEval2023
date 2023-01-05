# LIBRAIRIES
import os
import re
from pathlib import Path
from string import punctuation

import pandas as pd
import numpy as np

import spacy
from tqdm.auto import tqdm

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sentence_transformers import SentenceTransformer, util


# DATA -- A paramétrer si besoin
CUR_DIR = Path(__file__).parent
BASE_DIR = Path(CUR_DIR).parent
DATA_DIR = os.path.join(BASE_DIR, "Data")
files = [file for file in os.listdir(DATA_DIR) if ".csv" in file]
for file in files: 
    if file=="train.csv":
        TRAIN = os.path.join(DATA_DIR, file)
    elif file=="dev.csv":
        DEV = os.path.join(DATA_DIR, file)

nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# FONCTIONS
def process_text(text):
    """
    Traiter/Nettoyer un texte pour récupérer les lemmes, en enlevant les stopwords et les termes de ponctuation.
    """
    doc = nlp(text)
    lemmas = []
    for token in doc:
        ents = [ent.text for ent in doc.ents]
        if token.lemma_ not in set(stopwords) and token.lemma_ not in punctuation:
            if token.text not in ents:
                lemmas.append(token.lemma_)
    processed = ' '.join(token for token in lemmas)
    return processed

def evaluate_model(label, pred):
    """
    Créer, imprimer à la console et sauvegarder un rapport des scores des métriques d'évaluation
    """
    cr = metrics.classification_report(label, pred, zero_division=1, digits=3)
    print(cr, "\n")
    
    cr = metrics.classification_report(label, pred, zero_division=1, digits=3, output_dict=True)
    df_cr = pd.DataFrame(cr)
    filepath = os.path.join(Path.cwd(), "dev_classification_report.csv")
    df_cr.to_csv(filepath, sep=";", encoding="UTF-8")
    print(f"Métriques d'évaluation sauvegardés @ {filepath}.")
    

# CODE
tqdm.pandas(desc="Executing ")

# TRAIN
df = pd.read_csv(TRAIN, sep=";", encoding="UTF-8")

# Lemmatiser les textes d'entrée
print("Lemmatising...\n")
df["lemmatised"] = df["text"].progress_apply(process_text)
print("Lemmatisation completed.\n")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

df["embs"] = df["lemmatised"].progress_apply(model.encode)

X = np.array(df["embs"].tolist())
y = df["label"]

# Initialiser & Entraîner le modèle
neigh = KNeighborsClassifier(n_neighbors=55) # 55 meilleur k trouvé
neigh.fit(X, y)

# DEV
df_dev = pd.read_csv(DEV, sep=";", encoding="UTF-8")

print("Lemmatising...\n")
df_dev["lemmatised"] = df_dev["text"].progress_apply(process_text)
print("Lemmatisation completed.\n")

df_dev["embs"] = df_dev["lemmatised"].progress_apply(model.encode)

Y = np.array(df_dev["embs"].tolist())

# Faire les prédictions
df_dev["prediction"] = neigh.predict(Y)

# Evaluer les prédictions
evaluate_model(df_dev["label"], df_dev["prediction"])