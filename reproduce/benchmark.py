"""Print measured performance for synthetic data; no hard-coded results."""
from time import perf_counter
from nma.simulation import generate_shared_private
from nma.rsa import rsa_score
from nma.cca import CrossValidatedCCA
from nma.statistics import rsa_permutation_test

for n in (30, 60, 120, 240):
    a=generate_shared_private(n_samples=n,random_state=42).aligned
    results={}
    for name,call in (
        ('RSA',lambda:rsa_score(a,metric='euclidean')),
        ('CCA',lambda:CrossValidatedCCA(n_splits=5,pca_components=8).fit_evaluate(a)),
        ('RSA null x19',lambda:rsa_permutation_test(a,n_permutations=19,metric='euclidean')),
    ):
        t=perf_counter();call();results[name]=round(perf_counter()-t,4)
    print({'matched_units':n,'synthetic':True,'seconds':results})
