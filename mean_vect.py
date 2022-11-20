import os
from scipy import spatial
import sent2vec
import sys
from sent2vec.constants import PRETRAINED_VECTORS_PATH_WIKI, ROOT_DIR
from sent2vec.vectorizer import Vectorizer, BertVectorizer
from sent2vec.splitter import Splitter
import pandas as pd
import numpy
import re
vectorizer = Vectorizer()

df_train = pd.read_csv('RR_TRAIN.csv',sep='\t')


RR = ['PREAMBLE','FAC','RLC','ISSUE','ARG_PETITIONER','ARG_RESPONDENT','ANALYSIS','STA','PRE_RELIED','PRE_NOT_RELIED','RATIO','RPC','NONE']


split_pattern = '(?<=\.\?!)\\s*([A-Z])'
split_pattern_2 = '(?<=[ˆA-Z]{2}[\.\?!])\s+(?=[A-Z])'


sentences = []
mean = 0
count = 0

# CALCUL MOYENNE / CLASSE

for index, row in df_train.iterrows():
    if row['annotation_label'] == RR[10]: #CHANGE INDEX
        text = (row['annotation_text'])
        text = re.sub('''\\\\n''',' ',text)
        text = text.lower()
        text = re.split(split_pattern_2,text)
        sentences.append(text)
print(sentences)


for i in sentences:
    try:
        i = list(i)
        vectorizer.run(i)
        vectors = vectorizer.vectors
    except:
        pass
    mean += (numpy.mean(vectors))
    count += 1
    print(f'Etat : {count} / {len(sentences)}\n\n')
    #print(i)
    print(numpy.mean(vectors))

print(RR[10]) # CHANGE INDEX
print(mean/count)

