"""Regenerate every scientific figure; all results are synthetic, never empirical."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nma.simulation import generate_shared_private, make_temporal_lag
from nma.rsa import rsa_score, representational_dissimilarity
from nma.cca import CrossValidatedCCA
from nma.statistics import rsa_permutation_test
from nma.temporal import temporal_rsa
from nma.plotting import neon_style

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets';OUT.mkdir(exist_ok=True)

def export(fig,name):
    for ext in ('png','svg'):
        fig.savefig(OUT/(name+'.'+ext),dpi=200,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)

with neon_style():
    sim=generate_shared_private(n_samples=100,shared_strength=2.4,private_dim=2,noise=.2,random_state=42)
    a=sim.aligned
    from sklearn.decomposition import PCA
    x=PCA(n_components=2).fit_transform(a.X)
    y=PCA(n_components=2).fit_transform(a.Y)
    rsa=rsa_score(a,metric='euclidean')
    null=rsa_permutation_test(a,n_permutations=199,metric='euclidean',random_state=42)
    fig,axs=plt.subplots(1,4,figsize=(15,3.4))
    fig.suptitle('NEURO  /  MULTIMODAL ALIGNMENT',fontsize=18,fontweight='bold',x=.035,ha='left')
    axs[0].scatter(x[:,0],x[:,1],c=np.arange(len(x)),cmap='cool',s=13,alpha=.85)
    axs[0].set_title('01  ·  Modality A')
    axs[1].scatter(y[:,0],y[:,1],c=np.arange(len(y)),cmap='cool',s=13,alpha=.85)
    axs[1].set_title('02  ·  Modality B')
    rx=representational_dissimilarity(a.X,metric='euclidean')
    ry=representational_dissimilarity(a.Y,metric='euclidean')
    axs[2].scatter(rx[np.triu_indices(100,1)],ry[np.triu_indices(100,1)],s=2,c='#31f5e5',alpha=.2)
    axs[2].set_xlabel('A pairwise distance');axs[2].set_ylabel('B pairwise distance')
    axs[2].set_title(f'03  ·  Shared geometry  RSA={rsa:.2f}')
    axs[3].hist(null.null_scores,bins=22,color='#ba90ff',alpha=.85)
    axs[3].axvline(null.observed,color='#31f5e5',lw=2,label=f'Observed {rsa:.2f}')
    axs[3].legend(fontsize=8,frameon=False)
    axs[3].set_title(f'04  ·  Permutation null  p={null.pvalue:.3f}')
    for ax in axs:ax.grid(True,alpha=.12)
    fig.text(.035,.005,'SYNTHETIC DATA ONLY  •  FIXED SEED 42  •  NOT BIOLOGICAL EVIDENCE',fontsize=8,alpha=.75)
    fig.tight_layout(rect=(0,.045,1,.88));export(fig,'hero_synthetic')

    strengths=[0,.2,.5,1,2,3]
    scores=[]
    for strength in strengths:
        arr=[]
        for seed in range(8):
            b=generate_shared_private(n_samples=130,shared_strength=strength,noise=.2,random_state=seed).aligned
            arr.append(rsa_score(b,metric='euclidean'))
        scores.append(arr)
    z=np.asarray(scores)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(strengths,z.mean(axis=1),marker='o',lw=2,label='Mean across independent seeds')
    ax.fill_between(strengths,z.min(axis=1),z.max(axis=1),alpha=.2,label='Seed range, not CI')
    ax.set_xlabel('Shared signal amplitude (simulation setting)')
    ax.set_ylabel('RSA Spearman score')
    ax.set_title('Shared structure recovery  |  Synthetic benchmark')
    ax.legend(frameon=False);ax.grid(True,alpha=.15);fig.tight_layout();export(fig,'shared_structure')

    lagX,lagY,lag=make_temporal_lag(n_samples=85,noise=.10)
    temporal=temporal_rsa(lagX,lagY,metric='euclidean',n_permutations=199)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(temporal.times,temporal.scores,lw=2.3,label='Observed synthetic RSA')
    ax.axvline(lag,ls=':',color='#ffad5b',label='Injected synthetic peak')
    ax.axhline(np.quantile(temporal.null_max,.95),ls='--',color='#ba90ff',label='95th percentile max-|null|')
    ax.set_xlabel('Synthetic time sample');ax.set_ylabel('RSA score')
    ax.set_title('Temporal alignment with multiple-comparison control')
    ax.legend(frameon=False,fontsize=8);ax.grid(True,alpha=.12);fig.tight_layout();export(fig,'temporal_alignment')

    n=[20,40,80,160]
    valid=[];invalid=[]
    from sklearn.cross_decomposition import CCA
    for nobs in n:
        rng=np.random.default_rng(900+nobs)
        X=rng.normal(size=(nobs,20));Y=rng.normal(size=(nobs,20))
        fit=CCA(n_components=1,max_iter=2000).fit(X,Y)
        tx,ty=fit.transform(X,Y)
        invalid.append(np.corrcoef(tx[:,0],ty[:,0])[0,1])
        al=__import__('nma.data',fromlist=['AlignedDataset']).AlignedDataset(X,Y,range(nobs),'observation')
        valid.append(CrossValidatedCCA(n_components=1,n_splits=5,pca_components=10).fit_evaluate(al).overall_score)
    fig,ax=plt.subplots(figsize=(7.6,4))
    ax.plot(n,invalid,'o-',label='Invalid in-sample CCA')
    ax.plot(n,valid,'s-',label='Held-out CCA')
    ax.axhline(0,ls=':',color='#bdcfe2',lw=1)
    ax.set_xlabel('Matched independent simulated observations')
    ax.set_ylabel('First canonical correlation')
    ax.set_title('Why cross-validation matters  |  Independent modalities')
    ax.legend(frameon=False);ax.grid(True,alpha=.1);fig.tight_layout();export(fig,'cca_leakage')

print('Generated eight synthetic figure files (4 png + 4 svg) in',OUT)
