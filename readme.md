# Guide d'utilisation

Avant de pouvoir utiliser l'outil de tests de regex, il est conseillé de placer les fichiers originaux de corpus dans un répertoire `data` et  d'utiliser le parser `parser1.py` qui créera des fichiers CSV pour faciliter la tâche.
Pour tester les regex, il faut lancer `main.py` ; celui-ci récupère les motifs présents dans `regex_config.json` et les teste sur les ensembles de données spécifiés dans ce même fichier.
Concernant `main.py`, des paramètres peuvent être modifiés dans le header pour faciliter la prise en main : `print_false_positives`, `print_false_negatives` et `print_true_positive` peuvent prendre `True` ou `False` pour faciliter la visualisation des résultats.
Aussi, il est possible de modifier `excluded_tags` pour que le système ignore ou non certaines regex contenues dans `regex_config.json`, entre autres pour pouvoir en tester une sans avoir à réécrire tout le fichier `regex_config.json` ou attendre longuement qu'elles tournent toutes.
