# Aspects généraux

Nous avons pour objectif ici de participer aux tâches RR et L-NER de la campagne d'évaluation LegalEval 2013.

# RR

[...]

# L-NER

## Guide d'utilisation & préparation des données

Pour cette tâche, nous avons testé tout d'abord des expressions régulières, ce qui a donné des résultats modérément bons, ce qui a ensuite invité l'utilisation de CRFs.
Ces différents aspects sont détaillés ci-dessous, mais il faut avant tout préparer les données :
- après avoir mis les fichiers de corpus dans le répertoire `data`, lancer `parser1.py` pour transformer les données dans un format CSV
- pour séparer en TRAIN/DEV (pour l'entrainement des CRFs), lancez ensuite `corpus_splitter_v2.py`

Si vous utilisez les CRFs, d'autres fichiers seront générés automatiquement par le script de CRF, ceux-ci devraient s'autogérer.

## Regex

Avant de pouvoir utiliser l'outil de tests de regex, il est conseillé de placer les fichiers originaux de corpus dans un répertoire `data` et  d'utiliser le parser `parser1.py` qui créera des fichiers CSV pour faciliter la tâche.
Pour tester les regex, il faut lancer `main.py` ; celui-ci récupère les motifs présents dans `regex_config.json` et les teste sur les ensembles de données spécifiés dans ce même fichier.
Concernant `main.py`, des paramètres peuvent être modifiés dans le header pour faciliter la prise en main : `print_false_positives`, `print_false_negatives` et `print_true_positive` peuvent prendre `True` ou `False` pour faciliter la visualisation des résultats.
Aussi, il est possible de modifier `excluded_tags` pour que le système ignore ou non certaines regex contenues dans `regex_config.json`, entre autres pour pouvoir en tester une sans avoir à réécrire tout le fichier `regex_config.json` ou attendre longuement qu'elles tournent toutes.

## ESSAIS CRF (NOUVELLE VERSION)

La version la plus évoluée du système CRF que nous avons est `test_CRF3.py`.
Ce script se base sur `default_CRF_config.json` pour sa configuration.
Vous pouvez principalement renseigner :
- quelles _features_ prendre en considération (`relevant_features`, `irrelevant_features`)
- quel empan prendre en considération (`look_behind`, `look_ahead`)
- quel nom donner au système
- les différents paramètres de la librairie **sklearn--crfsuite**

Le script entraine un système (si `training = True`) puis calcule sa performance sur l'ensemble DEV selon différentes métriques (précision, rappel, F1, perfect/relaxed match).
Un fichier de configuration système est généré avec un nom similaire à celui que vous avez donné au système (`*_config.json`) ; cela permet de recharger exactement les bons paramètres si vous souhaitez recharger le système (par défaut, si vous laissez le nom de fichier du modèle dans le fichier de configuration, le script le rechargera tel qu'il a été sauvegardé).
Aussi, après chaque lancement, les prédictions réalisées sont mises en mémoire dans `latest_results.csv` (à la condition d'avoir la _feature_ 'text' de sélectionnée) si jamais l'on souhaite comparer en détails ce qui a été prédit ce qui aurait dû l'être.

## Essais CRF (ANCIENNE VERSION)

### Input/Output

`test_CRF.py` permet de tester différentes configurations de CRF/features sur la classification NER.
Renseignez un nom de fichier dans `model_path`, et le modèle généré y sera sauvegardé ; à ce sujet, `load_model`, `save_model`, et `training` peuvent être en `True` ou `False`.
À noter qu'il me fallait ici toute la base en TRAIN/DEV, j'ai donc fait un script alternatif de split (corpus_splitter_v2.py), je ne sais pas lequel on utilisera à la fin, je vous laisse me dire.

### TRAIN/DEV

Le script effectue (après entrainement ou non) une évaluation sur TRAIN, puis une évaluation sur DEV.
À ce sujet, `features_memory_train_path` et `features_memory_dev_path` sont pour sauvegarder les résultats de SpaCy sous la forme de CSV pour ne pas avoir à tout recalculer à chaque fois.
Avant les résultats par classe, les deux scores indiqués correspondent à l'accuracy toutes classes confondues (y compris NONE), et l'accuracy pour toutes les classes confondues sauf NONE.

### Features

Concernant le choix des features, `relevant_features` permet de renseigner les features que l'on souhaite garder quand on en génère un nouveau jeu avant d'être sauvegardé dans un CSV.
À l'inverse, `irrelevant_features` permet d'ignorer certains features en rechargeant le CSV (cela permet de faire différents tests rapidement sans avoir à tout refaire).
`look_behind` et `look_ahead` permettent d'indiquer combien de tokens avant et après prendre en compte (cela duplique en fait les features des tokens en question sur le token actuel, en renommant la feature en +n:nom_feature) ; je conseille de ne pas mettre plus de 2 et 2 si le nombre de features est élevé, pour des questions de consommation mémoire.

### Discussion

La meilleure configuration que j'ai eue (je l'ai sauvegardée avec les autres) a eu 0.93 en accuracy sur TRAIN et 0.65 sur DEV, avec un peu de travail on pourrait améliorer ça pour affiner tout/rajouter des epochs de training.
Cependant, avec beaucoup de features j'ai le ressenti que le système fait de l'overfitting sur les données d'entrainement.
Je pense qu'il faudrait donc soit diminuer le nombre de features à des features générales, soit faire des classifieurs spécifiques à certaines classes (surtout que si l'on fait certaines par regex, pas besoin de les prendre en compte dans le CRF ?).
Tout ceci est à discuter.