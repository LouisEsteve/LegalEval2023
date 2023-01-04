# LIBRAIRIES
import os
from pathlib import Path
from string import punctuation

from collections import defaultdict
import operator
import copy

import spacy
nlp = spacy.load("en_core_web_sm")

import pandas as pd

from tqdm.auto import tqdm

from sklearn import metrics


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
    return lemmas

def get_ngrams(tokens, n):
    """
    Découper une liste de tokens en n-grams, n à désigner.
    """
    temp = zip(*[tokens[i:] for i in range(0, n)])
    ngrams = [' '.join(ngram) for ngram in temp]
    return ngrams

def evaluate_class(lemmas, top_ngrams):
    """
    Faire des prédictions sur les textes d'entrée. Procédé :
    1. Pour chaque label, enumérer le nombre de fois que chacun des top 20 n-grams est apparu dans un texte
    2. Calculer pour chaque label, le nombre total des fois où ses top 20 n-grams sont apparus sur le nombre total de tous les n-grams apparus (même s'ils sont apparus plusieurs fois)
    3. Classer les labels en order décroissant et donner le label dont les n-grams sont les plus nombreux
    """
    ngram_list = copy.deepcopy(top_ngrams)
    for label, ngrams in ngram_list.items():
        for word in lemmas:
            if word in ngrams.keys():
                ngrams[word] += 1
    count_raw = defaultdict()
    for label, ngrams in ngram_list.items():
        count_raw[label] = sum(1 for item in ngrams.values() if item!=0)
    total = sum(count_raw.values())
    count = {k:round(v/total, 2) for k,v in count_raw.items() if total!=0}
    count = sorted(count, key=count.get, reverse=True)
    if count:
        prediction = count[0]
    else:
        prediction = "NONE"
    return prediction


# CODE
df = pd.read_csv(TRAIN, sep=";", encoding="UTF-8")
print(f"Learning from {TRAIN}\n")

tqdm.pandas(desc="Executing ")

# Lemmatiser les textes d'entrée
print("Lemmatising...\n")
df["lemmatised"] = df["text"].progress_apply(process_text)
print("Lemmatisation completed.\n")

# Obtenir les n-grams (Uni-grams by default)
df["ngrams"] = df["lemmatised"].progress_apply(get_ngrams, args=(1,))
print("N-grams obtained.\n")

print(df["ngrams"].head(10))

rr_labels = df["label"].unique().tolist()

# Obtenir top 20 n-grams de chaque label
ngram_counts = {x:defaultdict(lambda: 0) for x in rr_labels}
top_ngrams = {x:defaultdict(lambda: 0) for x in rr_labels}

for label, ddict in ngram_counts.items():
    print("Processing: ", label)
    ngram_list = df[df['label']==label].ngrams
    ngram_list = list(ngram_list)
    for sublist in ngram_list:
        for item in sublist:
            ddict[item] += 1
    sorted_d = sorted(ddict.items(), key=operator.itemgetter(1), reverse=True)
    top_20 = sorted_d[:20]
    top_ngrams[label] = {k:0 for k,v in top_20}
    print("\n")


# Evaluer les datasets

## TRAIN
# Evaluer les n-grams appris sur le TRAIN lui-même
# print("Evaluating the TRAIN dataset...\n")
# df["prediction"] = df["lemmatised"].progress_apply(evaluate_class, args=(top_ngrams,))
# output = os.path.join(CUR_DIR, "train_predictions.csv")
# df.to_csv(output, sep=";", encoding="UTF-8")

## DEV
# Evaluer les n-grams appris sur le DEV
print("Evaluating the DEV dataset...\n")
df_dev = pd.read_csv(DEV, sep=";", encoding="UTF-8")

# Lemmatiser les textes d'entrée
print("Lemmatising...\n")
df_dev["lemmatised"] = df_dev["text"].progress_apply(process_text)

# Obtenir les n-grams
df_dev["ngrams"] = df_dev["lemmatised"].progress_apply(get_ngrams, args=(1,))
print("N-grams obtained.\n")

print(f"Generating predictions for {DEV}...\n")
df_dev["prediction"] = df_dev["lemmatised"].progress_apply(evaluate_class, args=(top_ngrams,))

print("Predictions obtained for DEV.")
output = os.path.join(CUR_DIR, "dev_predictions.csv")
df_dev.to_csv(output, sep=";", encoding="UTF-8")

# Produire les métriques d'évaluation
classification_report = metrics.classification_report(df_dev['label'], df_dev['prediction'], zero_division=1, digits=3)
print(classification_report)
print("...\n")

classification_report = metrics.classification_report(df_dev['label'], df_dev['prediction'], zero_division=1, digits=3, output_dict=True)

df_cr = pd.DataFrame(classification_report)
filepath = os.path.join(CUR_DIR, "dev_classreport.csv")
df_cr.to_csv(filepath, sep=";", encoding="UTF-8")

print(f"Métriques d'évaluation sauvegardés @ {filepath}.")