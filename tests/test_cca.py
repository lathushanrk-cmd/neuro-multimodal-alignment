import numpy as np
import pytest
from nma.cca import CrossValidatedCCA
from nma.data import AlignedDataset
from nma.simulation import generate_shared_private

def test_true_shared():
    a=generate_shared_private(shared_strength=3,private_dim=1,noise=.05,n_samples=180).aligned
    r=CrossValidatedCCA(n_components=1,n_splits=5,pca_components=8).fit_evaluate(a)
    assert r.overall_score>.8

def test_independent_is_not_high():
    a=generate_shared_private(shared_dim=0,private_dim=4,n_samples=200,random_state=9).aligned
    r=CrossValidatedCCA(n_components=1,n_splits=5,pca_components=6).fit_evaluate(a)
    assert abs(r.overall_score)<.45

def test_invalid_splits(aligned):
    with pytest.raises(ValueError):CrossValidatedCCA(n_splits=100).fit_evaluate(aligned)

def test_grouped(aligned):
    groups=np.repeat(np.arange(20),5)
    r=CrossValidatedCCA(n_splits=5,pca_components=5).fit_evaluate(aligned,groups=groups)
    assert r.fold_correlations.shape==(5,1)

def test_overdimension(aligned):
    with pytest.raises(ValueError):CrossValidatedCCA(n_components=100).fit_evaluate(aligned)
