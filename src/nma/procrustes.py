"""Orthogonal Procrustes with strictly held-out evaluation."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.model_selection import KFold, GroupKFold
from .preprocessing import fitted_preprocessor

@dataclass
class ProcrustesResult:
    fold_rmse: np.ndarray
    fold_baseline_rmse: np.ndarray
    @property
    def mean_rmse(self): return float(np.mean(self.fold_rmse))
    @property
    def improvement_over_baseline(self):
        return float(1-np.mean(self.fold_rmse)/np.mean(self.fold_baseline_rmse))

class CrossValidatedProcrustes:
    def __init__(self, n_components=2, n_splits=5, random_state=42):
        self.n_components=n_components;self.n_splits=n_splits;self.random_state=random_state
    def fit_evaluate(self,aligned,groups=None):
        X,Y=aligned.X,aligned.Y
        if not 2<=self.n_splits<=len(X)//2: raise ValueError('Need >=2 rows per test fold')
        if groups is None:
            splits = KFold(self.n_splits,shuffle=True,random_state=self.random_state).split(X)
        else:
            g = np.asarray(groups)
            if g.ndim != 1 or len(g) != len(X) or len(np.unique(g)) < self.n_splits:
                raise ValueError('groups need >=n_splits independent groups')
            splits = GroupKFold(self.n_splits).split(X,Y,g)
        out=[];base=[]
        for tr,te in splits:
            if len(te) < 2:
                raise ValueError('Each held-out fold needs at least two observations')
            if not 1<=self.n_components<=min(len(tr)-1,X.shape[1],Y.shape[1]):
                raise ValueError('Invalid components')
            px=fitted_preprocessor(X[tr],self.n_components)
            py=fitted_preprocessor(Y[tr],self.n_components)
            xtr,xt=px.transform(X[tr]),px.transform(X[te])
            ytr,yt=py.transform(Y[tr]),py.transform(Y[te])
            R,_=orthogonal_procrustes(xtr,ytr)
            # Orthogonal fit cannot change singular values; scaling is NOT silently estimated.
            pred=xt@R
            out.append(float(np.sqrt(np.mean((pred-yt)**2))))
            base.append(float(np.sqrt(np.mean(yt**2))))
        return ProcrustesResult(np.array(out),np.array(base))

def recover_rotation(X,Y):
    X,Y=np.asarray(X,float),np.asarray(Y,float)
    if X.ndim!=2 or X.shape!=Y.shape: raise ValueError('Same shape expected')
    r,_=orthogonal_procrustes(X,Y)
    return r
