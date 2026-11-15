"""Run: python examples/01_ground_truth_simulation.py"""
from nma.simulation import generate_shared_private
from nma.rsa import rsa_score
from nma.statistics import rsa_permutation_test

sim = generate_shared_private(n_samples=160, shared_strength=2.5, noise=.1, random_state=42)
a = sim.aligned
print('Synthetic RSA:', round(rsa_score(a, metric='euclidean'), 3))
p = rsa_permutation_test(a, metric='euclidean', n_permutations=199, random_state=42)
print('Permutation p-value (synthetic):', p.pvalue)
