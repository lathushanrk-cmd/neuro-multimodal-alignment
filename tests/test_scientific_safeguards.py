"""Independent safeguards for pairing, block exchangeability, and group-level evaluation."""
import numpy as np
import pytest
from nma.data import AlignedDataset
from nma.simulation import generate_shared_private, make_temporal_lag
from nma.temporal import temporal_rsa
from nma.procrustes import CrossValidatedProcrustes
from nma.shared_private import crossmodal_predictability
from nma.cca import CrossValidatedCCA
from nma.rsa import rsa_score


def test_bad_independently_supplied_identifiers_do_not_silently_pair():
    x = np.arange(36.).reshape(12, 3)
    y = x[::-1].copy()
    ids = [f'trial-{i}' for i in range(12)]
    with pytest.raises(ValueError, match='exactly the same IDs'):
        AlignedDataset.from_mismatched_rows(x, y, ids, ids[:-1] + ['unmatched'], 'observation')


def test_matching_by_id_recovers_paired_data():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 6))
    perm = rng.permutation(len(x))
    ids = [f'unit-{i}' for i in range(len(x))]
    reordered = AlignedDataset.from_mismatched_rows(x, x[perm], ids, np.asarray(ids)[perm], 'observation')
    assert np.array_equal(reordered.X, reordered.Y)
    assert rsa_score(reordered, metric='euclidean') > .999


def test_missing_identifiers_rejected_before_pairing():
    x = np.ones((3, 4))
    with pytest.raises(ValueError, match='missing or blank'):
        AlignedDataset.from_mismatched_rows(x, x, ['a', None, 'c'], ['a', None, 'c'], 'observation')


def test_blocked_temporal_is_reproducible():
    x, y, lag = make_temporal_lag(n_samples=60, n_times=20, lag_index=8)
    blocks = np.repeat(np.arange(6), 10)
    a = temporal_rsa(x, y, metric='euclidean', blocks=blocks, n_permutations=29, random_state=4)
    b = temporal_rsa(x, y, metric='euclidean', blocks=blocks, n_permutations=29, random_state=4)
    assert np.array_equal(a.scores, b.scores)
    assert np.array_equal(a.null_max, b.null_max)
    assert np.array_equal(a.pvalues_fwer, b.pvalues_fwer)
    assert abs(np.argmax(a.scores) - lag) <= 2
    assert np.all((a.pvalues_fwer >= 1/30) & (a.pvalues_fwer <= 1))


def test_temporal_rejects_unpermutable_blocks():
    x, y, _ = make_temporal_lag(n_samples=18, n_times=9, lag_index=4)
    with pytest.raises(ValueError, match='No exchangeable'):
        temporal_rsa(x, y, blocks=np.arange(len(y)), n_permutations=5)


def test_temporal_rejects_bad_block_shape():
    x, y, _ = make_temporal_lag(n_samples=18, n_times=9, lag_index=4)
    with pytest.raises(ValueError, match='blocks'):
        temporal_rsa(x, y, blocks=[1,2], n_permutations=5)


def test_grouped_procrustes_runs_and_rejects_bad_groups():
    aligned = generate_shared_private(n_samples=60, random_state=2).aligned
    groups = np.repeat(np.arange(12), 5)
    output = CrossValidatedProcrustes(n_splits=4).fit_evaluate(aligned, groups=groups)
    assert len(output.fold_rmse) == 4
    assert np.isfinite(output.fold_rmse).all()
    with pytest.raises(ValueError, match='groups'):
        CrossValidatedProcrustes(n_splits=4).fit_evaluate(aligned, groups=np.zeros(60))


def test_grouped_shared_private_runs():
    aligned = generate_shared_private(n_samples=60,random_state=2).aligned
    result = crossmodal_predictability(aligned, groups=np.repeat(np.arange(12), 5), n_splits=4)
    assert np.isfinite(result.x_predictable_from_y)
    assert np.isfinite(result.y_predictable_from_x)


def test_grouped_cca_runs_and_rejects_invalid_split():
    aligned = generate_shared_private(n_samples=60,random_state=2).aligned
    groups = np.repeat(np.arange(12), 5)
    result = CrossValidatedCCA(n_splits=4).fit_evaluate(aligned,groups=groups)
    assert len(result.fold_correlations) == 4
    with pytest.raises(ValueError, match='groups'):
        CrossValidatedCCA(n_splits=4).fit_evaluate(aligned,groups=np.zeros(60))


def test_zero_shared_structure_is_not_forced_high():
    data = generate_shared_private(n_samples=180, n_x=10, n_y=12, shared_dim=3,
                                   shared_strength=0, noise=.3, random_state=33)
    score = rsa_score(data.aligned, metric='euclidean')
    assert abs(score) < .25


def test_high_shared_structure_positive_rsa():
    data = generate_shared_private(n_samples=200, n_x=10, n_y=12, shared_dim=3,
                                   shared_strength=5, noise=.05, random_state=33)
    assert rsa_score(data.aligned, metric='euclidean') > .7
