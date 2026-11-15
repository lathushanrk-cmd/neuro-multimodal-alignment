import numpy as np
import pytest
from nma.procrustes import recover_rotation,CrossValidatedProcrustes
from nma.data import AlignedDataset

def test_known_rotation():
    rng=np.random.default_rng(8);x=rng.normal(size=(100,4));q,_=np.linalg.qr(rng.normal(size=(4,4)))
    r=recover_rotation(x,x@q)
    assert np.linalg.norm(r-q)<1e-8

def test_cv_rotation():
    rng=np.random.default_rng(33);x=rng.normal(size=(120,3));q,_=np.linalg.qr(rng.normal(size=(3,3)))
    a=AlignedDataset(x,x@q,range(120),'observation')
    r=CrossValidatedProcrustes(n_components=3).fit_evaluate(a)
    assert r.mean_rmse<.1

def test_bad_dimension(aligned):
    with pytest.raises(ValueError):CrossValidatedProcrustes(n_components=200).fit_evaluate(aligned)
