"""Co-régression COREG (Zhou & Li, 2005) pour `modeles.ipynb`, partie 7.

Principe : deux régresseurs kNN, chacun sur une vue disjointe des entrées
(ici composition chimique contre paramètres de soudage + traitement
thermique), se pseudo-étiquettent mutuellement les lignes où la cible
n'est pas mesurée. À chaque itération, chaque vue ajoute la ligne non
labellisée qui fait le plus baisser son erreur sur ses voisins labellisés
(critère Δ > 0 de l'article), étiquetée par l'autre vue.

Anti-fuite : la réserve non labellisée est filtrée par article à chaque
pli (interne comme externe) — un article du pli de test ne fournit ni
cible ni ligne non labellisée. Voir `evaluer_semi`.

Usage (dans le notebook, de même structure que les autres familles) :
    from coreg import COREGRegressor, ajuster_semi, evaluer_semi, recherche_interne
    def coreg(X):
        return COREGRegressor()
    ajouter_semi("COREG (réglé)", coreg, {"k1": [5, 15], "k2": [5, 15]})
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupKFold, ParameterGrid

from ml_dataset import HEADER_RE, col_num, make_preprocessor

N_VUE_COMPO_MAX = 21  # cols 1-21 : composition ; le reste : procédé + TT


def vues_par_defaut(X):
    """Vue 1 = composition (cols 1-21), vue 2 = le reste des entrées.

    Repli : moitié/moitié si une vue serait vide (garde-fou, sans objet ici).
    """
    vue1 = [c for c in X.columns if HEADER_RE.match(c) and col_num(c) <= N_VUE_COMPO_MAX]
    vue2 = [c for c in X.columns if c not in vue1]
    if not vue1 or not vue2:
        milieu = len(X.columns) // 2
        vue1, vue2 = list(X.columns[:milieu]), list(X.columns[milieu:])
    return vue1, vue2


def _knn_predire(Z_app, y_app, Z_req, k):
    """Moyenne des k plus proches voisins (poids uniformes, euclidien)."""
    k = max(1, min(k, Z_app.shape[0]))
    d = pairwise_distances(Z_req, Z_app, metric="euclidean")
    idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    return y_app[idx].mean(axis=1)


class COREGRegressor(BaseEstimator, RegressorMixin):
    """COREG avec deux kNN sur deux vues, chaque vue avec sa préparation.

    Chaque vue a sa propre préparation (`make_preprocessor` sur ses
    colonnes, réapprise à chaque `fit` sur les seules lignes labellisées),
    car un `ColumnTransformer` unique mélangerait les vues. Sans réserve
    (`X_non_lab=None`), repli supervisé : moyenne des deux kNN.
    """

    def __init__(self, vue1=None, vue2=None, k1=5, k2=5, n_iter=30,
                 taille_reserve=100, random_state=0):
        self.vue1 = vue1
        self.vue2 = vue2
        self.k1 = k1
        self.k2 = k2
        self.n_iter = n_iter
        self.taille_reserve = taille_reserve
        self.random_state = random_state

    def fit(self, X, y, X_non_lab=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("COREGRegressor attend X en DataFrame (noms de colonnes pour les vues).")
        v1, v2 = (list(self.vue1), list(self.vue2)) if self.vue1 is not None else vues_par_defaut(X)
        self.vue1_, self.vue2_ = v1, v2
        y_all = np.asarray(y, dtype=float).ravel()
        self.prep1_ = make_preprocessor(X[v1])
        self.prep2_ = make_preprocessor(X[v2])
        Z1 = np.asarray(self.prep1_.fit_transform(X[v1]), dtype=float)
        Z2 = np.asarray(self.prep2_.fit_transform(X[v2]), dtype=float)
        self.k1_eff_ = max(1, min(self.k1, len(y_all)))
        self.k2_eff_ = max(1, min(self.k2, len(y_all)))
        y1, y2 = y_all.copy(), y_all.copy()
        self.n_iter_ = 0
        if X_non_lab is not None and len(X_non_lab) and len(y_all) >= 3:
            rng = np.random.RandomState(self.random_state)
            U1 = np.asarray(self.prep1_.transform(X_non_lab[v1]), dtype=float)
            U2 = np.asarray(self.prep2_.transform(X_non_lab[v2]), dtype=float)
            for _ in range(self.n_iter):
                i1, Z1, y1 = self._selection(Z1, y1, self.k1_eff_, U1, U2,
                                             Z2, y2, self.k2_eff_, rng)
                if i1 is not None:
                    U1, U2 = np.delete(U1, i1, axis=0), np.delete(U2, i1, axis=0)
                i2, Z2, y2 = self._selection(Z2, y2, self.k2_eff_, U2, U1,
                                             Z1, y1, self.k1_eff_, rng)
                if i2 is not None:
                    U1, U2 = np.delete(U1, i2, axis=0), np.delete(U2, i2, axis=0)
                if i1 is None and i2 is None:
                    break
                self.n_iter_ += 1
        self.Z1_, self.y1_, self.Z2_, self.y2_ = Z1, y1, Z2, y2
        self.n_ajouts_1_, self.n_ajouts_2_ = len(y1) - len(y_all), len(y2) - len(y_all)
        return self

    def _selection(self, Z, y, k, U, Uo, Zo, yo, ko, rng):
        """Une étape de co-entraînement pour une vue.

        Renvoie (indice global dans U, Z étendu, y étendu) ; l'indice est
        None si aucun candidat ne fait baisser l'erreur (Δ ≤ 0).
        """
        if not len(U):
            return None, Z, y
        tirage = rng.choice(len(U), size=min(self.taille_reserve, len(U)), replace=False)
        pseudo = _knn_predire(Zo, yo, Uo[tirage], ko)
        u, delta = self._meilleur_candidat(Z, y, k, U[tirage], pseudo)
        if u is None or delta <= 0:
            return None, Z, y
        return tirage[u], np.vstack([Z, U[tirage][u]]), np.append(y, pseudo[u])

    @staticmethod
    def _meilleur_candidat(Z, y, k, P, pseudo):
        """Candidat maximisant la baisse d'erreur LOO sur ses k voisins labellisés."""
        L = len(Z)
        keff = max(1, min(k, L - 1)) if L > 1 else 1
        Dll = pairwise_distances(Z, Z, metric="euclidean")
        masque = Dll.copy()
        np.fill_diagonal(masque, np.inf)
        idx_loo = np.argpartition(masque, kth=keff, axis=1)[:, :keff + 1]
        pred_loo = y[idx_loo[:, :keff]].mean(axis=1)
        Dpl = pairwise_distances(P, Z, metric="euclidean")
        voisins = np.argpartition(Dpl, kth=min(keff, L) - 1, axis=1)[:, :min(keff, L)]
        meilleur, delta_max = None, 0.0
        for iu in range(len(P)):
            Ns = voisins[iu]
            base = ((y[Ns] - pred_loo[Ns]) ** 2).sum()
            neuf = 0.0
            for n in Ns:
                cand_idx = idx_loo[n]
                dd = np.append(Dll[n, cand_idx], Dpl[iu, n])
                sel = np.argpartition(dd, kth=keff - 1)[:keff]
                vals = np.append(y[cand_idx], pseudo[iu])[sel]
                neuf += (y[n] - vals.mean()) ** 2
            if base - neuf > delta_max:
                delta_max, meilleur = base - neuf, iu
        return meilleur, delta_max

    def predict(self, X):
        Z1 = np.asarray(self.prep1_.transform(X[self.vue1_]), dtype=float)
        Z2 = np.asarray(self.prep2_.transform(X[self.vue2_]), dtype=float)
        return (_knn_predire(self.Z1_, self.y1_, Z1, self.k1_eff_)
                + _knn_predire(self.Z2_, self.y2_, Z2, self.k2_eff_)) / 2


def recherche_interne(modele_base, grille, X, y, groupes, X_non_lab, groupes_non_lab,
                       n_plis=5, graine=0):
    """Meilleurs hyperparamètres par validation croisée interne groupée.

    La réserve est refiltrée par article dans chaque pli interne : un
    article du pli de validation interne ne fournit pas de non-labellisé.
    """
    cv = GroupKFold(n_splits=n_plis)
    meilleur, meilleur_mse = None, np.inf
    for params in ParameterGrid(grille):
        mses = []
        for train, val in cv.split(X, y, groupes):
            g_train = set(groupes.iloc[train].tolist())
            if X_non_lab is not None:
                masque = groupes_non_lab.isin(g_train).to_numpy()
                Xu = X_non_lab[masque] if masque.any() else None
            else:
                Xu = None
            m = clone(modele_base).set_params(**params)
            m.fit(X.iloc[train], y.iloc[train], X_non_lab=Xu)
            pred = m.predict(X.iloc[val])
            mses.append(mean_squared_error(y.iloc[val], pred))
        mse = float(np.mean(mses))
        if mse < meilleur_mse:
            meilleur, meilleur_mse = params, mse
    return dict(meilleur)


def ajuster_semi(modele_base, params, X, y, X_non_lab):
    """Clone, règle les hyperparamètres et entraîne (réserve déjà filtrée)."""
    m = clone(modele_base).set_params(**params)
    return m.fit(X, y, X_non_lab=X_non_lab)


def evaluer_semi(modele_base, grille, X, y, groupes, X_non_lab, groupes_non_lab,
                 n_plis=5, graine=0):
    """Prédictions hors pli par article + mesures, comme `evaluer` (§1).

    Renvoie (pred, mesures, choix_plis) avec les mêmes clés de mesures
    ("RMSE", "RMSE écart-type entre plis", "MAE", "R²") pour réutiliser
    `tableau_complet`, `comparer`, `hyperparametres`, `figure_predit_reel`.
    """
    cv = GroupKFold(n_splits=n_plis)
    pred = pd.Series(np.nan, index=y.index, dtype=float)
    rmse_plis, choix_plis = [], []
    for train, test in cv.split(X, y, groupes):
        g_train = set(groupes.iloc[train].tolist())
        if X_non_lab is not None:
            masque = groupes_non_lab.isin(g_train).to_numpy()
            Xu = X_non_lab[masque] if masque.any() else None
            gu = groupes_non_lab[masque] if masque.any() else None
        else:
            Xu, gu = None, None
        params = recherche_interne(modele_base, grille, X.iloc[train], y.iloc[train],
                                   groupes.iloc[train], Xu, gu, n_plis, graine)
        choix_plis.append(params)
        m = ajuster_semi(modele_base, params, X.iloc[train], y.iloc[train], Xu)
        pred.iloc[test] = m.predict(X.iloc[test])
        rmse_plis.append(mean_squared_error(y.iloc[test], pred.iloc[test]) ** 0.5)
    mesures = {
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "RMSE écart-type entre plis": float(np.std(rmse_plis)),
        "MAE": mean_absolute_error(y, pred),
        "R²": r2_score(y, pred),
    }
    return pred, mesures, choix_plis
