"""Bounded checks of real plotting context and explicit methodological assumptions."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
from nma.plotting import neon_style
from nma.simulation import generate_shared_private
from nma.statistics import rsa_permutation_test, independent_bootstrap


def test_neon_plot_context_does_not_leak_global_style(tmp_path):
    original = matplotlib.rcParams['figure.facecolor']
    with neon_style():
        fig, ax = plt.subplots()
        assert matplotlib.rcParams['figure.facecolor'] == '#081220'
        ax.plot([0, 1], [0, 1], label='Synthetic trend')
        ax.legend()
        fig.savefig(tmp_path/'test_style.png')
        plt.close(fig)
    assert matplotlib.rcParams['figure.facecolor'] == original
    assert (tmp_path/'test_style.png').stat().st_size > 200


def test_restricted_rsa_permutation_rejects_no_exchangeable_units():
    simulated = generate_shared_private(n_samples=12)
    with pytest.raises(ValueError, match='No permutable'):
        rsa_permutation_test(simulated.aligned, blocks=np.arange(12), n_permutations=5)


def test_rsa_permutation_reproducible_under_blocks():
    simulated = generate_shared_private(n_samples=40, random_state=3)
    groups = np.repeat(np.arange(4), 10)
    a = rsa_permutation_test(simulated.aligned, blocks=groups, metric='euclidean', n_permutations=19, random_state=7)
    b = rsa_permutation_test(simulated.aligned, blocks=groups, metric='euclidean', n_permutations=19, random_state=7)
    assert a.observed == b.observed
    assert np.array_equal(a.null_scores, b.null_scores)
    assert a.pvalue == b.pvalue
    assert 1/20 <= a.pvalue <= 1


def test_bootstrap_repeatability_and_bounds():
    x = np.array([3., 5., 6., 6., 7., 10.])
    a = independent_bootstrap(x, n_bootstrap=199, random_state=22)
    assert a == independent_bootstrap(x, n_bootstrap=199, random_state=22)
    assert a[0] <= np.mean(x) <= a[1]
