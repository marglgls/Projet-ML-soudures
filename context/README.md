# Fichiers de contexte

Métadonnées du jeu de données MAP_DATA_WELD, construites à partir des fichiers téléchargés dans `data/welddb/` (voir la partie 1.2 du notebook `data_preprocessing.ipynb`). Elles ne contiennent pas les données.

- `columns.json` : une entrée par colonne (1 à 44). Libellés et unités tirés de `welddb.info` ; type, rôle et descriptions ajoutés par l'équipe, en partie générés avec une IA ; statistiques (`missing`, `observed`) calculées sur `welddb.data`, en partie par les scripts `update_columns.py` et `add_outliers.py` de `preprocessing/`.
- `data_explantions.md` : liste des colonnes recopiée de `welddb.info`.
