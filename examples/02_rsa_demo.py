"""Run: python examples/02_rsa_demo.py"""
import numpy as np
from nma.rsa import representational_dissimilarity, rsa_score
rng=np.random.default_rng(5)
x=rng.normal(size=(40,10))
y=x@rng.normal(size=(10,15))+.1*rng.normal(size=(40,15))
print('Example synthetic RDM shape:', representational_dissimilarity(x,'euclidean').shape)
print('Synthetic RSA:',round(rsa_score(x,y,metric='euclidean'),3))
