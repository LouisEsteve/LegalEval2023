# Aspects généraux

Nous avons pour objectif ici de participer aux tâches RR et L-NER de la campagne d'évaluation LegalEval 2013.

# RR

[...]

# L-NER

Version Python utilisée : `3.10.1`

## Guide d'utilisation & préparation des données

Afin de préparer les données :
- après avoir mis les fichiers de corpus dans le répertoire `data`, lancer `parser1.py` pour transformer les données dans un format CSV
```sh
python parser1.py
```
- pour séparer en TRAIN/DEV (pour l'entrainement des CRFs), lancez ensuite `corpus_splitter_v2.py`
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
- entraine un système (si `training = True`)
- calcule sa performance sur l'ensemble DEV selon différentes métriques (précision, rappel, F1, perfect/relaxed match)
- génère un fichier de configuration système avec un nom similaire à celui que vous avez donné au système (`*_config.json`) ; cela permet de recharger exactement les bons paramètres si vous souhaitez recharger le système (par défaut, si vous laissez le nom de fichier du modèle dans le fichier de configuration, le script le rechargera tel qu'il a été sauvegardé).
- garde une trace des prédictions réalisées dans `latest_results.csv` (à la condition d'avoir la _feature_ 'text' sélectionnée) si jamais l'on souhaite comparer en détails ce qui a été prédit ce qui aurait dû l'être.

### Prédiction sur le corpus

Un script alternatif fait spécifiquement pour la structure JSON de la tâche est executable par
```sh
python L-NER_CRF.py -f mon_fichier_de_corpus.json
```
pour cibler un fichier spécifique, ou par
```sh
python L-NER_CRF.py -d chemin/vers/corpus
```
pour cibler tout un répertoire.
Plusieurs `-f` et `-d` peuvent être mis dans un même lancement de script, y compris conjointement.

Ce script réutilise des fonctions présentes dans `L-NER_CRF_train.py`.
En sortie, sont générés automatiquement un fichier `*_output.json` pour chaque fichier traité, avec les nouvelles annotations ajoutées.


## Regex

Avant de pouvoir utiliser l'outil de tests de regex, il est conseillé de placer les fichiers originaux de corpus dans un répertoire `data` et  d'utiliser le parser `parser1.py` qui créera des fichiers CSV pour faciliter la tâche.
Pour tester les regex, il faut lancer `main.py` ; celui-ci récupère les motifs présents dans `regex_config.json` et les teste sur les ensembles de données spécifiés dans ce même fichier.
Concernant `main.py`, des paramètres peuvent être modifiés dans le header pour faciliter la prise en main : `print_false_positives`, `print_false_negatives` et `print_true_positive` peuvent prendre `True` ou `False` pour faciliter la visualisation des résultats.
Aussi, il est possible de modifier `excluded_tags` pour que le système ignore ou non certaines regex contenues dans `regex_config.json`, entre autres pour pouvoir en tester une sans avoir à réécrire tout le fichier `regex_config.json` ou attendre longuement qu'elles tournent toutes.
