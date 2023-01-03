import pandas as pd
import numpy as np
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement DF
df_train = pd.read_csv('RR_TRAIN_alt.csv', encoding='UTF-8', sep='\t')
df_test = pd.read_csv('RR_DEV_alt.csv', encoding='UTF-8', sep='\t')

# LabelEncoder: étiquettes --> int
le = LabelEncoder()
df_train['annotation_label'] = le.fit_transform(df_train['annotation_label'])
df_test['annotation_label'] = le.transform(df_test['annotation_label'])

train_text = df_train['annotation_text'].fillna(' ')
test_text = df_test['annotation_text'].fillna(' ')

# Pipeline + choix du classifieur
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('classifier', LogisticRegression()) # ACCURACY : 0.55
    #('classifier', MultinomialNB()) # ACCURACY : 0.53
])

# Hyperparamètres
param_grid = {
    'vectorizer__max_features': [1000, 2000, 3000, 4000, 5000],
    #'classifier__alpha': [0.01, 0.1, 1.0] #MultinomialNB())
    'classifier__C': [0.01, 0.1, 1.0] #LogisticRegression())
}

# Validation croisée
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(train_text, df_train['annotation_label'])

print(f"Meilleurs paramètres: {grid_search.best_params_}")


# Chargement d'un modèle (si disponible)
try:
    with open("best_model.pkl", "rb") as f:
        grid_search = pickle.load(f)
        print("Modèle précédemment enregistré chargé avec succès.")
except:
    print("Aucun modèle précédemment enregistré trouvé. Utilisation du modèle entraîné.")
    # Enregistrement du meilleur modèle dans le fichier "best_model.pkl"
    with open("best_model.pkl", "wb") as f:
        pickle.dump(grid_search, f)
        print("Meilleur modèle enregistré dans le fichier 'best_model.pkl'.")


# Predictions
y_pred = grid_search.predict(test_text)


# Predictions validation croisée
scores = cross_val_score(grid_search, train_text, df_train['annotation_label'], cv=5)

# Récupération des noms des étiquettes
labels = le.inverse_transform(np.arange(len(le.classes_)))

# Matrice de confusion
confusion_mat = confusion_matrix(df_test['annotation_label'], y_pred)

# Affichage de la matrice de confusion
print("Matrice de confusion :")
print(confusion_mat)

# Calcul du rapport de classification
classification_report = classification_report(df_test['annotation_label'], y_pred, target_names=labels)

# Affichage du rapport de classification

print(classification_report)

# Calcul de l'accuracy
accuracy = accuracy_score(df_test['annotation_label'], y_pred)

# Affichage de l'accuracy
print(f"\nAccuracy : {accuracy:.3f}")


# Matrice de confusion avec seaborn
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap="crest", xticklabels=labels, yticklabels=labels)
plt.ylabel('Valeurs réelles')
plt.xlabel('Valeurs prédites')
plt.title("Matrice de confusion")

# Enregistrement de l'image
plt.savefig("confusion_matrix.png")