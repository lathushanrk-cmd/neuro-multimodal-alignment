"""Run: python examples/03_crossvalidated_cca.py"""
from nma.simulation import generate_shared_private
from nma.cca import CrossValidatedCCA
sim=generate_shared_private(n_samples=150,shared_strength=3,private_dim=2,random_state=3)
result=CrossValidatedCCA(n_components=1,pca_components=8).fit_evaluate(sim.aligned)
print('Held-out synthetic canonical correlations by fold:',result.fold_correlations[:,0].round(3).tolist())
print('Mean:',round(result.overall_score,3))
