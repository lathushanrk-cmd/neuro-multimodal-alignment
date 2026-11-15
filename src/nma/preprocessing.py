"""Strict fit-on-training transformations and fold-safe PCA."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

def fitted_preprocessor(X_train, n_components=None):
    X = np.asarray(X_train, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or not np.isfinite(X).all():
        raise ValueError('Training data must be a finite 2D array with >=3 rows')
    if n_components is not None:
        if not 1 <= n_components <= min(X.shape[0]-1, X.shape[1]):
            raise ValueError('PCA components must be <= min(n_train-1, n_features)')
        pipeline = make_pipeline(StandardScaler(), PCA(n_components=n_components, svd_solver='full'))
    else:
        pipeline = make_pipeline(StandardScaler())
    pipeline.fit(X)
    return pipeline
