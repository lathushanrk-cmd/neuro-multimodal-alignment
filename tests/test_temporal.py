import numpy as np
import pytest
from nma.simulation import make_temporal_lag
from nma.temporal import temporal_rsa

def test_known_lag():
    X,Y,lag=make_temporal_lag(n_samples=75,noise=.07)
    res=temporal_rsa(X,Y,metric='euclidean',n_permutations=29)
    assert abs(np.argmax(res.scores)-lag)<=3
    assert len(res.pvalues_fwer)==X.shape[0]

def test_bad_shape():
    with pytest.raises(ValueError):temporal_rsa(np.ones((5,3)),np.ones((5,3)))

def test_bad_time():
    X,Y,_=make_temporal_lag(n_times=10,lag_index=5)
    with pytest.raises(ValueError):temporal_rsa(X,Y,times=np.ones(10))
