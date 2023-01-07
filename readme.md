# Aspects généraux

Nous avons pour objectif ici de participer aux tâches RR et L-NER de la campagne d'évaluation LegalEval 2023.

# RR

## Vecteurs bow et classification par régression logistique.

### Prérequis
- `pickle5`
- `sklearn`
- `pandas`
- `nltk`
- `matplotlib`
- `seaborn`
- `numpy`

### Données d'entrée
Les données d'entraînement [RR_TRAIN_alt.csv](RR/Data/RR_TRAIN_alt.csv) et les données test [RR_DEV_alt.csv](RR/Data/RR_DEV_alt.csv) sont disponsibles dans le dossier [Data](RR/Data).

### Fonctionnement et utilisation
Le script `classifieur.py` permet d'entraîner le modèle sur un jeu de données d'entraînement, puis de tester sur un jeu de données de test. Par défaut, le classifieur utilisé est `LogisticRegression` car c'est celui qui donne les meilleurs résultats. Le nombre de plis pour la validation croisée est paramétrée à 5.
Des hyperparamètres ont été ajoutés et l'objet `GridSearchCV` permet de trouver la combinaison qui donne les meilleurs performances. Le meilleur modèle entraîné est enregistré dans un fichier au format `.plk`. Une fois le modèle entraîné, il est utilisé pour prédire les étiquettes du jeu de données de test. A la fin de l'éxécution du script, un tableau au format csv est enregistré. Ce fichier contient les colonnes `annotation_text`, `annotation label` et `predicted_label`. C'est ce ficier qui sera utilisé comme données d'entrée pour le scipt `analyse.py`

### Analyse des résultats
Le script `analyse.py` permet l'affichage d'un rapport de classification et d'une matrice de confusion. Il affiche égalemennt une matrice de confusion au format png. Enfin, la fonction `overlap.py` permet de retourner les classes ayant reçu le plus d'annotations incorrectes.

## Vecteurs sémantiques et classification par kNN

### Prérequis
- `sentence_transformers`
- `pandas` 
- `numpy`
- `sklearn`
-  `tqdm`
-  `pathlib`

### Données d'entrée
Vous pouvez générer les fichiers de données nécessaires à partir des fichiers bruts en exécutant le script [data_extraction.py](RR/Data/data_extraction.py). Le script prend les fichiers bruts fournis par les organisateurs et les convertit en format .csv.

### Utilisation
Le script [sent_trf.py](RR/vecteurs/sentence_transformers/sent_trf.py) nettoie, lemmatise et vectorise les données du fichier [train.csv](RR/Data/train.csv). Un modèle de k-Nearest Neighbours est entraîné avec ces vecteurs et labels comme donnée d'entrée. Puis, le script pré-traite un nouveau jeu de données DEV [dev.csv](RR/Data/dev.csv) et génère des prédictions.

A la sortie, un fichier avec les prédictions [dev_pred.csv](RR/vecteurs/sentence_transformers/dev_pred.csv) et un rapport de classification [dev_classreport.csv](RR/vecteurs/sentence_transformers/dev_classification_report.csv) seront produits.

# L-NER

## Prérequis

Version Python recommandée : `3.10.9`

Veuillez noter qu'il n'est pas possible d'utiliser Python `3.11.0` et au-delà du fait que `pycrfsuite`, utilisé par `sklearn_crfsuite`, n'est plus compatible à partir de cette version.

Librairies :
- `joblib`
- `json`
- `pandas`
- `seqeval`
- `sklearn`
- `sklearn_crfsuite`
- `spacy`

Pour plus de détails concernant les versions recommandées, veuillez vous référer à [requirements.txt](/requirements.txt).

## Guide d'utilisation & préparation des données

Afin de préparer les données :
- après avoir mis les fichiers de corpus dans le répertoire `/data`, lancez [parser1.py](/parser1.py) pour transformer les données dans un format CSV
```sh
python parser1.py
```
- si vous n'avez pas de DEV à votre disposition, pour séparer en TRAIN/DEV (pour l'entrainement des CRFs), lancez ensuite [corpus_splitter_v2.py](/corpus_splitter_v2.py)
```sh
python corpus_splitter_v2.py
```

Si vous utilisez les CRFs, d'autres fichiers seront générés automatiquement par le script de CRF, ceux-ci devraient s'autogérer.

## CRF

### Entrainement

Le script pour entrainer un CRF est exécutable par
```sh
python L_NER_CRF_train.py
```
Ce script se base sur [L_NER_CRF_default_config.json](/L_NER_CRF_default_config.json) pour sa configuration.
Vous pouvez spécifier une configuration alternative en modifiant `config_path`.
Dans un fichier de configuration, vous pouvez principalement renseigner :
- quelles _features_ prendre en considération (`relevant_features`, `irrelevant_features`)
- quel empan prendre en considération (`look_behind`, `look_ahead`)
- quel nom donner au système
- les différents paramètres de la librairie **sklearn--crfsuite**

Le script
- entraine un système (si `training == True`)
- calcule sa performance sur l'ensemble DEV selon différentes métriques (précision, rappel, F1, perfect/relaxed match)
- génère un fichier de configuration système avec un nom similaire à celui que vous avez donné au système (`*_config.json`) ; cela permet de recharger exactement les bons paramètres si vous souhaitez recharger le système (par défaut, si vous laissez le nom de fichier du modèle dans le fichier de configuration, le script le rechargera tel qu'il a été sauvegardé).
- garde une trace des prédictions réalisées dans [latest_results.csv](/latest_results.csv) (à la condition d'avoir la _feature_ 'text' sélectionnée) si jamais l'on souhaite comparer en détails ce qui a été prédit ce qui aurait dû l'être.

### Prédiction sur le corpus

Un script alternatif, [L_NER_CRF.py](/L_NER_CRF.py), fait spécifiquement pour la structure JSON de la tâche est disponible.
Pour l'exécuter sur un seul fichier :
```sh
python L_NER_CRF.py --f mon_fichier_de_corpus.json
```
ou
```sh
python L_NER_CRF.py --file mon_fichier_de_corpus.json
```


Pour l'exécuter sur un dossier complet :
```sh
python L_NER_CRF.py --d chemin/vers/corpus
```
ou
```sh
python L_NER_CRF.py --directory chemin/vers/corpus
```
Plusieurs `--f` et `--d` peuvent être mis dans un même lancement de script, y compris conjointement.

Ce script réutilise des fonctions présentes dans [L_NER_CRF_train.py](/L_NER_CRF_train.py).
En sortie, est généré automatiquement un fichier `*_OUTPUT.json` pour chaque fichier traité, avec les nouvelles annotations ajoutées.


## Regex (ancienne méthode)

/!\ POUR LES REGEX UTILISEES EN POST-TRAITEMENT DE CRF, REFEREZ VOUS À LA FONCTION `post_processing` DE [L_NER_CRF_train.py](/L_NER_CRF_train.py)  ET `post_processing_from_raw_offsets` DE [L_NER_CRF.py](/L_NER_CRF.py) /!\

Avant de pouvoir utiliser l'outil de tests de regex, il est conseillé de placer les fichiers originaux de corpus dans un répertoire `/data` et  d'utiliser le parser [parser1.py](/parser1.py) qui créera des fichiers CSV pour faciliter la tâche.
Pour tester les regex, il faut lancer [main.py](/L_NER_old/regex/main.py) ; celui-ci récupère les motifs présents dans [regex_config.json](/L_NER_old/regex/regex_config.json) et les teste sur les ensembles de données spécifiés dans ce même fichier.
Concernant [main.py](/L_NER_old/regex/main.py), des paramètres peuvent être modifiés dans le header pour faciliter la prise en main : `print_false_positives`, `print_false_negatives` et `print_true_positive` peuvent prendre `True` ou `False` pour faciliter la visualisation des résultats.
Aussi, il est possible de modifier `excluded_tags` pour que le système ignore ou non certaines regex contenues dans [regex_config.json](/L_NER_old/regex/regex_config.json), entre autres pour pouvoir en tester une sans avoir à réécrire tout le fichier [regex_config.json](/L_NER_old/regex/regex_config.json) ou attendre longuement qu'elles tournent toutes.
