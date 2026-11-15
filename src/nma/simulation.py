"""Synthetic paired data with known shared/private factor construction."""

from dataclasses import dataclass
import numpy as np
from .data import AlignedDataset


@dataclass
class SimulatedPair:
    aligned: AlignedDataset
    shared_latent: np.ndarray
    x_private: np.ndarray
    y_private: np.ndarray
    parameters: dict


def generate_shared_private(
    n_samples=200,
    n_x=24,
    n_y=30,
    shared_dim=3,
    private_dim=3,
    shared_strength=1.0,
    noise=0.15,
    random_state=42,
):
    if (
        n_samples < 6
        or min(n_x, n_y) < 1
        or shared_dim < 0
        or private_dim < 0
        or (shared_dim + private_dim) == 0
    ):
        raise ValueError("Invalid simulation dimensions")
    if noise < 0 or shared_strength < 0:
        raise ValueError("Strength/noise cannot be negative")
    rng = np.random.default_rng(random_state)
    shared = rng.normal(size=(n_samples, shared_dim))
    xp = rng.normal(size=(n_samples, private_dim))
    yp = rng.normal(size=(n_samples, private_dim))

    def mix(latent, nf):
        if latent.shape[1] == 0:
            return np.zeros((n_samples, nf))
        return latent @ (rng.normal(size=(latent.shape[1], nf)) / np.sqrt(max(1, latent.shape[1])))

    X = shared_strength * mix(shared, n_x) + mix(xp, n_x) + noise * rng.normal(size=(n_samples, n_x))
    Y = shared_strength * mix(shared, n_y) + mix(yp, n_y) + noise * rng.normal(size=(n_samples, n_y))
    aligned = AlignedDataset(X, Y, [f"unit-{i:04d}" for i in range(n_samples)], pairing="condition")
    return SimulatedPair(
        aligned,
        shared,
        xp,
        yp,
        dict(
            n_samples=n_samples,
            n_x=n_x,
            n_y=n_y,
            shared_dim=shared_dim,
            private_dim=private_dim,
            shared_strength=shared_strength,
            noise=noise,
            random_state=random_state,
        ),
    )


def make_temporal_lag(
    n_samples=80, n_times=35, n_x_features=8, n_y_features=9, lag_index=12, noise=0.15, random_state=42
):
    if not 0 <= lag_index < n_times:
        raise ValueError("Lag index outside grid")
    rng = np.random.default_rng(random_state)
    shared = rng.normal(size=(n_samples, 3))
    Y = shared @ rng.normal(size=(3, n_y_features)) + noise * rng.normal(size=(n_samples, n_y_features))
    t = np.arange(n_times)
    profile = np.exp(-0.5 * ((t - lag_index) / 0.85) ** 2)
    mixing = rng.normal(size=(3, n_x_features))
    X = np.stack(
        [
            profile[i] * (shared @ mixing) + noise * rng.normal(size=(n_samples, n_x_features))
            for i in range(n_times)
        ],
        axis=0,
    )
    return X, Y, lag_index
