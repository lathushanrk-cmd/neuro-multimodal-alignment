import numpy as np
import pytest
from nma.statistics import rsa_permutation_test,independent_bootstrap
from nma.data import AlignedDataset

def test_permutation_signal():
    rng=np.random.default_rng(7);x=rng.normal(size=(30,5))
    a=AlignedDataset(x,x,range(30),'observation')
    z=rsa_permutation_test(a,n_permutations=99,random_state=3)
    assert z.pvalue<=.02

def test_permutation_deterministic(aligned):
    a=rsa_permutation_test(aligned,19)
    b=rsa_permutation_test(aligned,19)
    assert np.array_equal(a.null_scores,b.null_scores)

def test_bootstrap():
    assert independent_bootstrap(np.arange(20),n_bootstrap=100,random_state=0)==independent_bootstrap(np.arange(20),n_bootstrap=100,random_state=0)

def test_invalid_bootstrap():
    with pytest.raises(ValueError):independent_bootstrap([3])

def test_invalid_blocks(aligned):
    with pytest.raises(ValueError):rsa_permutation_test(aligned,blocks=np.arange(len(aligned.X)))
