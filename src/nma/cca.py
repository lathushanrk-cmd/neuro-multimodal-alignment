"""Nested fit/train transformations and held-out CCA; never report train correlation."""

from dataclasses import dataclass
import numpy as np
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold, GroupKFold
from .preprocessing import fitted_preprocessor


@dataclass
class CCAResult:
    fold_correlations: np.ndarray
    mean_correlations: np.ndarray
    fold_sizes: tuple

    @property
    def overall_score(self):
        return float(np.nanmean(self.fold_correlations[:, 0]))


class CrossValidatedCCA:
    def __init__(self, n_components=1, n_splits=5, pca_components=None, random_state=42):
        self.n_components = n_components
        self.n_splits = n_splits
        self.pca_components = pca_components
        self.random_state = random_state

    def fit_evaluate(self, aligned, groups=None):
        X, Y = aligned.X, aligned.Y
        n = len(X)
        if not 2 <= self.n_splits <= n // 2:
            raise ValueError("Each test fold needs at least two samples")
        if not 1 <= self.n_components <= min(X.shape[1], Y.shape[1]):
            raise ValueError("Too many CCA components")
        if groups is None:
            splits = KFold(self.n_splits, shuffle=True, random_state=self.random_state).split(X)
        else:
            if np.asarray(groups).ndim != 1 or len(groups) != n or len(np.unique(groups)) < self.n_splits:
                raise ValueError("groups must be a 1D label per observation with >=n_splits groups")
            splits = GroupKFold(self.n_splits).split(X, Y, groups)
        scores = []
        sizes = []
        for tr, te in splits:
            if len(te) < 2:
                raise ValueError("Each test fold must contain at least two independent rows")
            nx = (
                min(self.pca_components, X.shape[1], len(tr) - 1) if self.pca_components is not None else None
            )
            ny = (
                min(self.pca_components, Y.shape[1], len(tr) - 1) if self.pca_components is not None else None
            )
            px = fitted_preprocessor(X[tr], nx)
            py = fitted_preprocessor(Y[tr], ny)
            xt, xv = px.transform(X[tr]), px.transform(X[te])
            yt, yv = py.transform(Y[tr]), py.transform(Y[te])
            if self.n_components > min(xt.shape[1], yt.shape[1], len(tr) - 1):
                raise ValueError("CCA dimension exceeds fitted feature or training-sample rank")
            model = CCA(n_components=self.n_components, max_iter=2000)
            model.fit(xt, yt)
            a, b = model.transform(xv, yv)
            fold = []
            for i in range(self.n_components):
                if np.std(a[:, i]) < 1e-12 or np.std(b[:, i]) < 1e-12:
                    fold.append(np.nan)
                else:
                    fold.append(float(pearsonr(a[:, i], b[:, i]).statistic))
            scores.append(fold)
            sizes.append(len(te))
        scores = np.array(scores)
        return CCAResult(scores, np.nanmean(scores, axis=0), tuple(sizes))
