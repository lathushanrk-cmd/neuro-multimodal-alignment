"""Valid restricted permutation utilities and independent-unit bootstrap."""
from dataclasses import dataclass
import numpy as np
from .rsa import rsa_score

@dataclass
class PermutationResult:
    observed: float
    null_scores: np.ndarray
    pvalue: float
    alternative: str

def rsa_permutation_test(aligned, n_permutations=999, random_state=42, blocks=None, alternative='greater',metric='correlation'):
    if n_permutations<1: raise ValueError('Require at least one permutation')
    if alternative not in ('greater','less','two-sided'): raise ValueError('Invalid alternative')
    obs=rsa_score(aligned, metric=metric)
    rng=np.random.default_rng(random_state);null=[]
    groups=np.zeros(len(aligned.X),dtype=int) if blocks is None else np.asarray(blocks)
    if len(groups)!=len(aligned.X): raise ValueError('blocks length mismatch')
    group_indices=[np.where(groups==g)[0] for g in np.unique(groups)]
    if all(len(g)<2 for g in group_indices): raise ValueError('No permutable units in blocks')
    for _ in range(n_permutations):
        ix=np.arange(len(groups))
        for indices in group_indices: ix[indices]=rng.permutation(indices)
        null.append(rsa_score(aligned.X,aligned.Y[ix],metric=metric))
    null=np.asarray(null)
    if alternative=='greater':count=np.count_nonzero(null>=obs)
    elif alternative=='less':count=np.count_nonzero(null<=obs)
    else:count=np.count_nonzero(abs(null)>=abs(obs))
    return PermutationResult(obs,null,(count+1)/(n_permutations+1),alternative)

def independent_bootstrap(values, statistic=np.mean, n_bootstrap=1000, random_state=42):
    arr=np.asarray(values,dtype=float)
    if arr.ndim!=1 or len(arr)<2 or not np.isfinite(arr).all() or n_bootstrap<1:
        raise ValueError('Need >=2 finite independent estimates')
    rng=np.random.default_rng(random_state)
    stats=np.array([statistic(arr[rng.integers(0,len(arr),size=len(arr))]) for _ in range(n_bootstrap)])
    return tuple(np.percentile(stats,[2.5,97.5]))
