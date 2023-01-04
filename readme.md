# Aspects généraux

Nous avons pour objectif ici de participer aux tâches RR et L-NER de la campagne d'évaluation LegalEval 2013.

# RR

## Vecteurs sémantiques

### Prérequis
- `sent2vec`
- `pandas` 
- `numpy` 

### Données d'entrée
Les données d'entraînement [train.csv](RR/Data/train.csv) et les données test [dev.csv](RR/Data/dev.csv) sont disponsibles dans le dossier [Data](RR/Data).

### Utilisation
Le script [ngrams.py](RR/ngrams/ngrams.py) entraîne un système de n-grams en utilisant le fichier [train.csv](RR/Data/train.csv) comme donnée d'entrée. Une fois avoir appris les n-grams des données d'entrée, il fait des prédictions sur les données test [dev.csv](RR/Data/dev.csv). 
A la sortie, un fichier avec les prédictions [dev_predictions.csv](RR/ngrams/dev_predictions.csv) et un rapport de classification [dev_classreport.csv](RR/ngrams/dev_classreport.csv) seront produits.


## Classifieur

### Prérequis
- `pickle5`
- `sklearn`
- `pandas`
- `nltk`
- `matplotlib`
- `seaborn`
- `numpy`

### Données d'entrée
Les données d'entraînement [train.csv](RR/Data/train.csv) et les données test [dev.csv](RR/Data/dev.csv) sont disponsibles dans le dossier [Data](RR/Data).

### Utilisation
Le script [ngrams.py](RR/ngrams/ngrams.py) entraîne un système de n-grams en utilisant le fichier [train.csv](RR/Data/train.csv) comme donnée d'entrée. Une fois avoir appris les n-grams des données d'entrée, il fait des prédictions sur les données test [dev.csv](RR/Data/dev.csv). 
A la sortie, un fichier avec les prédictions [dev_predictions.csv](RR/ngrams/dev_predictions.csv) et un rapport de classification [dev_classreport.csv](RR/ngrams/dev_classreport.csv) seront produits.

## N-grams

### Prérequis
- `SpaCy==3.2` (la version actuelle `3.4.4` peut ne pas être compatible avec le modèle `en_core_web_sm` deployé)  
- `sklearn`
- `pandas`
-  `tqdm`

### Données d'entrée
Les données d'entraînement [train.csv](RR/Data/train.csv) et les données test [dev.csv](RR/Data/dev.csv) sont disponsibles dans le dossier [Data](RR/Data). Le cas échéant, vous pouvez générer de nouveau les fichiers de données nécessaires à partir des fichiers brutes en exécutant le script [data_extraction.py](RR/Data/data_extraction.py).

### Utilisation
Le script [ngrams.py](RR/ngrams/ngrams.py) entraîne un système de n-grams en utilisant le fichier [train.csv](RR/Data/train.csv) comme donnée d'entrée. Une fois avoir appris les n-grams des données d'entrée, il fait des prédictions sur les données test [dev.csv](RR/Data/dev.csv). 
A la sortie, un fichier avec les prédictions [dev_predictions.csv](RR/ngrams/dev_predictions.csv) et un rapport de classification [dev_classreport.csv](RR/ngrams/dev_classreport.csv) seront produits.


# L-NER

Version Python recommandée : `3.10.9`

Veuillez noter qu'il n'est pas possible d'utiliser Python `3.11.0` et au-delà du fait que `pycrfsuite`, utilisé par `sklearn_crfsuite`, n'est plus compatible à partir de cette version.

## Guide d'utilisation & préparation des données

Afin de préparer les données :
- après avoir mis les fichiers de corpus dans le répertoire `data`, lancez `parser1.py` pour transformer les données dans un format CSV
```sh
python parser1.py
```
- si vous n'avez pas de DEV à votre disposition, pour séparer en TRAIN/DEV (pour l'entrainement des CRFs), lancez ensuite `corpus_splitter_v2.py`
```sh
python corpus_splitter_v2.py
```

Si vous utilisez les CRFs, d'autres fichiers seront générés automatiquement par le script de CRF, ceux-ci devraient s'autogérer.

## CRF

### Entrainement

Le script pour entrainer un CRF est exécutable par
```sh
python L-NER_CRF_train.py
```
Ce script se base sur `L-NER_CRF_default_config.json` pour sa configuration.
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
- garde une trace des prédictions réalisées dans `latest_results.csv` (à la condition d'avoir la _feature_ 'text' sélectionnée) si jamais l'on souhaite comparer en détails ce qui a été prédit ce qui aurait dû l'être.

### Prédiction sur le corpus

Un script alternatif fait spécifiquement pour la structure JSON de la tâche est disponible.
Pour l'exécuter sur un seul fichier :
```sh
python L-NER_CRF.py --f mon_fichier_de_corpus.json
```
ou
```sh
python L-NER_CRF.py --file mon_fichier_de_corpus.json
```


Pour l'exécuter sur un dossier complet :
```sh
python L-NER_CRF.py --d chemin/vers/corpus
```
ou
```sh
python L-NER_CRF.py --directory chemin/vers/corpus
```
Plusieurs `--f` et `--d` peuvent être mis dans un même lancement de script, y compris conjointement.

Ce script réutilise des fonctions présentes dans `L-NER_CRF_train.py`.
En sortie, est généré automatiquement un fichier `*_OUTPUT.json` pour chaque fichier traité, avec les nouvelles annotations ajoutées.


## Regex (ancienne méthode)

/!\ POUR LES REGEX UTILISEES EN POST-TRAITEMENT DE CRF, REFEREZ VOUS À LA FONCTION `post_traitement` DE `L_NER_CRF_train.py` /!\

Avant de pouvoir utiliser l'outil de tests de regex, il est conseillé de placer les fichiers originaux de corpus dans un répertoire `data` et  d'utiliser le parser `parser1.py` qui créera des fichiers CSV pour faciliter la tâche.
Pour tester les regex, il faut lancer `main.py` ; celui-ci récupère les motifs présents dans `regex_config.json` et les teste sur les ensembles de données spécifiés dans ce même fichier.
Concernant `main.py`, des paramètres peuvent être modifiés dans le header pour faciliter la prise en main : `print_false_positives`, `print_false_negatives` et `print_true_positive` peuvent prendre `True` ou `False` pour faciliter la visualisation des résultats.
Aussi, il est possible de modifier `excluded_tags` pour que le système ignore ou non certaines regex contenues dans `regex_config.json`, entre autres pour pouvoir en tester une sans avoir à réécrire tout le fichier `regex_config.json` ou attendre longuement qu'elles tournent toutes.
