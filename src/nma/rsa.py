"""Representational similarity; RDMs are computed over matched observations."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def representational_dissimilarity(X, metric="correlation"):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or not np.isfinite(X).all():
        raise ValueError("RDM input must be a finite n>=3 by features matrix")
    if metric not in ("correlation", "euclidean"):
        raise ValueError("Supported distance metrics: correlation, euclidean")
    if metric == "correlation" and np.any(np.std(X, axis=1) < 1e-12):
        raise ValueError("Correlation distance is undefined for constant observation vectors")
    return squareform(pdist(X, metric=metric))


def rsa_score(X, Y=None, metric="correlation"):
    if Y is None:
        X, Y = X.X, X.Y
    a = representational_dissimilarity(X, metric)
    b = representational_dissimilarity(Y, metric)
    if a.shape != b.shape:
        raise ValueError("RDMs must have matched units")
    idx = np.triu_indices(len(a), k=1)
    v1, v2 = a[idx], b[idx]
    if np.std(v1) < 1e-12 or np.std(v2) < 1e-12:
        raise ValueError("RSA undefined for constant dissimilarities")
    return float(spearmanr(v1, v2).statistic)
