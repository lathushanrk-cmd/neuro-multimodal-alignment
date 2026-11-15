import numpy as np
import pytest
from nma.rsa import rsa_score,representational_dissimilarity

def test_identical():
    rng=np.random.default_rng(12);x=rng.normal(size=(25,7))
    assert np.isclose(rsa_score(x,x),1)

def test_rdm_symmetric():
    x=np.arange(24).reshape(6,4)
    r=representational_dissimilarity(x,'euclidean')
    assert np.allclose(r,r.T)
    assert np.allclose(np.diag(r),0)

def test_constant_row():
    with pytest.raises(ValueError):representational_dissimilarity(np.ones((4,5)))

def test_bad_metric():
    with pytest.raises(ValueError):representational_dissimilarity(np.ones((4,2)),'unknown')

def test_simulation_alignment(aligned):
    assert rsa_score(aligned,metric='euclidean')>0.05
