# Main APIs

- `nma.data.AlignedDataset(X, Y, sample_ids=..., pairing=...)`: validated matched data and ID reordering.
- `nma.rsa.rsa_score` and `representational_dissimilarity`: **descriptive** RSA / RDMs.
- `nma.cca.CrossValidatedCCA.fit_evaluate(aligned, groups=None)`: held-out, optional grouped CCA.
- `nma.procrustes.CrossValidatedProcrustes.fit_evaluate(aligned, groups=None)`: grouped held-out rotational alignment.
- `nma.shared_private.crossmodal_predictability(aligned, groups=None)`: grouped linear cross-modal prediction.
- `nma.temporal.temporal_rsa(X_time, Y, times=None, blocks=None)`: timewise RSA and max-statistic null.
- `nma.statistics.rsa_permutation_test` and `independent_bootstrap`: conditional inferential procedures.
- `nma.simulation.generate_shared_private` and `make_temporal_lag`: seeded synthetic ground truths.

Read method docstrings and tests for full arguments and limitations. Pre-release APIs may change.
