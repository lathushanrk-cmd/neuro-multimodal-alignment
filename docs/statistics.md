# Statistical inference and dependence

**Descriptive RSA:** `rsa_score` compares the rank order of matched-unit pairwise dissimilarities. This calculation itself is not cross-validated.

**Permutation tests:** RSA permutes the correspondence between X and Y. The analyst must justify why shuffled labels are exchangeable. `blocks` restricts permutations to same-block observations in `rsa_permutation_test` and `temporal_rsa`. For time-resolved analysis, the maximum absolute null across tested times yields familywise corrected p-values. This does not fix every form of repeated-measure dependence or confounding.

**Cross-validation:** CCA, Procrustes and shared/private predictability support group labels that hold out distinct groups. Folds still need enough independent observations. Scaling and optional PCA are fitted using training samples only.

**Bootstrap:** `independent_bootstrap` assumes observations supplied are independent estimates; it is not a general hierarchical bootstrap for repeated trial samples.
