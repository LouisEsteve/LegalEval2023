# LIBRAIRIES
import os
import sys
from collections import defaultdict

import spacy
nlp = spacy.load("en_core_web_sm")

import pandas as pd


# DATA -- A paramétrer si besoin
CUR_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
BASE_DIR = os.path.dirname(CUR_DIR)
DATA_DIR = BASE_DIR + os.sep + "Data"
files = [file for file in os.listdir(DATA_DIR) if ".csv" in file]
for file in files:
    if file=="train.csv":
        TRAIN = DATA_DIR + os.sep + file
    elif file=="dev.csv":
        DEV = DATA_DIR + os.sep + file


stopwords = nlp.Defaults.stop_words

# FONCTIONS
def process_text(text):
    text = text.lower()
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if token.lemma_ not in set(stopwords) and token.lemma_!=""]
    return lemmas

# def get_ngrams(text, n):
#     """
#     Découper un texte en tokens, enlever les stopwords et renvoyer des n-grams. N à désigner.
#     """
#     temp = zip(*[token[i:] for i in range(0, n)])
#     ngrams = [' '.join(ngram) for ngram in temp]
#     return ngrams

# def count_ngrams(list_obj):
#     ngram_counts = {x:defaultdict(lambda: 0) for x in RR_types}
    
#     for label, ddict in ngram_counts.items():
#         # print('Processing:', label)
#         if list_obj:
#             for item in list_obj:
#                 ddict[item] += 1
#     return ngram_counts


# CODE
df = pd.read_csv(TRAIN, sep=";", encoding="UTF-8")
print(f"En cours de traitement : {TRAIN}\n")
# print(df_TRAIN.head(10))

df["lemmatised"] = df["text"].apply(process_text)

print(df[["text", "lemmatised"]].head(10))


