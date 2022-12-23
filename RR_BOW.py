import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier

df_train = pd.read_csv('RR_TRAIN_alt.csv',encoding='UTF-8',sep='\t')
df_test = pd.read_csv('RR_DEV_alt.csv',encoding='UTF-8',sep='\t')

df_train['annotation_label'].replace(['PREAMBLE','FAC','RLC','ISSUE','ARG_PETITIONER','ARG_RESPONDENT','ANALYSIS','STA','PRE_RELIED','PRE_NOT_RELIED','RATIO','RPC','NONE'],[0,1,2,3,4,5,6,7,8,9,10,11,12],inplace=True)
df_test['annotation_label'].replace(['PREAMBLE','FAC','RLC','ISSUE','ARG_PETITIONER','ARG_RESPONDENT','ANALYSIS','STA','PRE_RELIED','PRE_NOT_RELIED','RATIO','RPC','NONE'],[0,1,2,3,4,5,6,7,8,9,10,11,12],inplace=True)


train_text = df_train['annotation_text'].fillna(' ')
test_text = df_test['annotation_text'].fillna(' ')


'''documents = []

stemmer = WordNetLemmatizer()

for sen in range(0, len(train_text)):

    document = str(train_text[sen])
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    document = document.lower()
    #document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)'''


vectorizer = TfidfVectorizer(max_features=5538,stop_words=stopwords.words('english'))
X_train = vectorizer.fit_transform(train_text).toarray()
X_test = vectorizer.fit_transform(test_text).toarray()

y_train = df_train['annotation_label']
y_test = df_test['annotation_label']

#classifier = MultinomialNB() # 29
#classifier = GaussianNB() # 31
classifier = LogisticRegression() # 32


classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred)) 
