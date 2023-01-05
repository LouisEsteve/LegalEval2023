# LIBRAIRIES
import os
import re
from pathlib import Path
from string import punctuation

import pandas as pd
import numpy as np

import spacy
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer, util


# DATA -- A paramétrer si besoin
CUR_DIR = Path(__file__).parent
BASE_DIR = os.path.dirname(Path(__file__).parent)
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

# CODE
df = pd.read_csv(TRAIN, sep=";", encoding="UTF-8")

# Lemmatiser les textes d'entrée
print("Lemmatising...\n")
df["lemmatised"] = df["text"].progress_apply(process_text)
print("Lemmatisation completed.\n")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sents = df["lemmatised"].values

embs = model.encode(sents)


