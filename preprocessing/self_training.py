"""Auto-apprentissage (self-training) adossé à la forêt aléatoire (`modeles.ipynb`, §7.6).

Principe : une forêt est ajustée sur les lignes labellisées, prédit la
réserve non labellisée, puis les lignes les plus sûres sont réinjectées
dans l'entraînement avec leur prédiction comme cible, et on recommence.
Confiance = désaccord inter-arbres (écart-type des prédictions des
arbres) : à chaque itération on ajoute les plus confiants (écart-type
sous la moitié de l'écart-type labellisé, dans la limite de la moitié
de la réserve restante), 3 itérations au plus. Si rien ne passe le
seuil, rien n'est ajouté — contrairement au COREG qui remplit ses 30
ajouts même quand les vues sont mauvaises (transfert négatif sur le
Charpy, §7.2).

Anti-fuite : comme pour le COREG, la réserve est filtrée par article à
chaque pli via `evaluer_semi`. Sans réserve (`X_non_lab=None`), repli
supervisé : simple forêt aléatoire.

Usage (même structure que les autres familles) :
    from self_training import SelfTrainingForestRegressor
    def foret_auto(X):
        return SelfTrainingForestRegressor(random_state=GRAINE)
    ajouter_semi("forêt auto-apprentissage (réglée)", foret_auto, GRILLE_AUTO)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

from ml_dataset import make_preprocessor


class SelfTrainingForestRegressor(BaseEstimator, RegressorMixin):
    """Forêt aléatoire + pseudo-étiquetage itératif des plus confiants."""

    def __init__(self, n_estimators=300, min_samples_leaf=1, max_features=1 / 3,
                 n_iter=3, part_pool=0.5, seuil_ecart=0.5, random_state=0, n_jobs=-1):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_iter = n_iter
        self.part_pool = part_pool
        self.seuil_ecart = seuil_ecart
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _foret(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, random_state=self.random_state,
            n_jobs=self.n_jobs)

    def fit(self, X, y, X_non_lab=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SelfTrainingForestRegressor attend X en DataFrame.")
        y_all = np.asarray(y, dtype=float).ravel()
        seuil = self.seuil_ecart * float(np.std(y_all))
        # Sans standardisation, comme les autres modèles à base d'arbres (§1).
        self.prep_ = make_preprocessor(X).set_params(num__scale="passthrough")
        Z = np.asarray(self.prep_.fit_transform(X), dtype=float)
        self.n_ajoutes_ = []
        if X_non_lab is not None and len(X_non_lab):
            P = np.asarray(self.prep_.transform(X_non_lab), dtype=float)
            Za, ya = Z, y_all.copy()
            for _ in range(self.n_iter):
                if not len(P):
                    break
                foret = self._foret().fit(Za, ya)
                T = np.asarray([t.predict(P) for t in foret.estimators_])
                ecarts = T.std(axis=0)
                eligibles = np.where(ecarts <= seuil)[0]
                if not len(eligibles):
                    break
                k = min(max(1, int(len(P) * self.part_pool)), len(eligibles))
                elus = eligibles[np.argsort(ecarts[eligibles], kind="stable")[:k]]
                Za = np.vstack([Za, P[elus]])
                ya = np.append(ya, T.mean(axis=0)[elus])
                garde = np.ones(len(P), dtype=bool)
                garde[elus] = False
                P = P[garde]
                self.n_ajoutes_.append(int(len(elus)))
        else:
            Za, ya = Z, y_all
        self.rf_ = self._foret().fit(Za, ya)
        return self

    def predict(self, X):
        return self.rf_.predict(np.asarray(self.prep_.transform(X), dtype=float))
