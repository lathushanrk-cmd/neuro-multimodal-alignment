"""Descriptive model-dependent shared/private variance; no biological identity claim."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold
from .preprocessing import fitted_preprocessor

@dataclass
class SharedPrivateResult:
    x_predictable_from_y: float
    y_predictable_from_x: float
    explanation: str='Held-out cross-modal linear predictability; not uniquely biological shared variance.'

def crossmodal_predictability(aligned,n_splits=5,alpha=10.,random_state=42,groups=None):
    X,Y=aligned.X,aligned.Y
    if not 2<=n_splits<=len(X)//2 or alpha<=0: raise ValueError('Invalid split count or ridge alpha')
    if groups is None:
        splits = KFold(n_splits,shuffle=True,random_state=random_state).split(X)
    else:
        g = np.asarray(groups)
        if g.ndim != 1 or len(g) != len(X) or len(np.unique(g)) < n_splits:
            raise ValueError('groups need >=n_splits independent groups')
        splits = GroupKFold(n_splits).split(X,Y,g)
    xx=[];yy=[]
    for tr,te in splits:
        if len(te) < 2:
            raise ValueError('Each held-out fold needs at least two observations')
        px=fitted_preprocessor(X[tr]);py=fitted_preprocessor(Y[tr])
        xtr,xt=px.transform(X[tr]),px.transform(X[te]);ytr,yt=py.transform(Y[tr]),py.transform(Y[te])
        mx=Ridge(alpha=alpha).fit(ytr,xtr)
        my=Ridge(alpha=alpha).fit(xtr,ytr)
        # Return pooled MSE explained against a training-mean predictor (zero after training scaling).
        xx.append((np.sum((xt-mx.predict(yt))**2),np.sum(xt**2)))
        yy.append((np.sum((yt-my.predict(xt))**2),np.sum(yt**2)))
    return SharedPrivateResult(1-sum(a for a,b in xx)/sum(b for a,b in xx),
                               1-sum(a for a,b in yy)/sum(b for a,b in yy))
