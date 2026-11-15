"""Time-resolved cross-modal RSA with a max-statistic familywise null."""

from dataclasses import dataclass
import numpy as np
from .rsa import rsa_score


@dataclass
class TemporalResult:
    scores: np.ndarray
    null_max: np.ndarray
    pvalues_fwer: np.ndarray
    times: np.ndarray


def temporal_rsa(
    X_time, Y, times=None, metric="correlation", n_permutations=199, random_state=42, blocks=None
):
    # X_time convention: time x matched-units x features.
    X = np.asarray(X_time, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 3 or Y.ndim != 2 or X.shape[1] != Y.shape[0]:
        raise ValueError("Expected X_time[t, matched_unit, feature] and Y[matched_unit,feature]")
    if n_permutations < 1:
        raise ValueError("Need positive permutation count")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Temporal input must contain only finite feature values")
    groups = np.zeros(len(Y), dtype=int) if blocks is None else np.asarray(blocks)
    if groups.ndim != 1 or len(groups) != len(Y):
        raise ValueError("blocks must be a 1D label per matched unit")
    if np.any(np.asarray([v is None or str(v).lower() == "nan" for v in groups])):
        raise ValueError("Missing block labels")
    unit_groups = [np.flatnonzero(groups == key) for key in np.unique(groups)]
    if not any(len(ix) >= 2 for ix in unit_groups):
        raise ValueError("No exchangeable units within blocks")
    if times is None:
        times = np.arange(len(X))
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or len(times) != len(X) or np.any(np.diff(times) <= 0):
        raise ValueError("Times must be strictly increasing and match time dimension")
    observed = np.array([rsa_score(x, Y, metric=metric) for x in X])
    rng = np.random.default_rng(random_state)
    maxabs = []
    for _ in range(n_permutations):
        perm = np.arange(len(Y))
        for ix in unit_groups:
            perm[ix] = rng.permutation(ix)
        vals = np.array([rsa_score(x, Y[perm], metric=metric) for x in X])
        maxabs.append(np.max(np.abs(vals)))
    maxabs = np.asarray(maxabs)
    p = (1 + np.sum(maxabs[:, None] >= np.abs(observed)[None, :], axis=0)) / (n_permutations + 1)
    return TemporalResult(observed, maxabs, p, times)
