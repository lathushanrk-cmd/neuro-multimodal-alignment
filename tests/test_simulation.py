import numpy as np
import pytest
from nma.simulation import generate_shared_private
from nma.rsa import rsa_score
from nma.shared_private import crossmodal_predictability
from nma.preprocessing import fitted_preprocessor

def test_reproducible():
    a=generate_shared_private(random_state=10)
    b=generate_shared_private(random_state=10)
    assert np.array_equal(a.aligned.X,b.aligned.X)

def test_zero_shared_near_null():
    a=generate_shared_private(n_samples=220,shared_dim=0,noise=.1)
    assert abs(rsa_score(a.aligned,metric='euclidean'))<.22

def test_private_prediction():
    a=generate_shared_private(n_samples=150,shared_strength=3,private_dim=1)
    score=crossmodal_predictability(a.aligned,alpha=15.)
    assert score.x_predictable_from_y>.35

def test_training_only_scaler():
    X=np.array([[0.,0.],[1.,1.],[2.,2.],[100.,100.]])
    p=fitted_preprocessor(X[:3]);out=p.transform(X[3:])
    assert out.min()>50

def test_invalid_strength():
    with pytest.raises(ValueError):generate_shared_private(noise=-1)
