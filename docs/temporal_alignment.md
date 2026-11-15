# Temporal RSA — plain English

`temporal_rsa(X_time, Y, times, metric="euclidean", n_permutations=199, blocks=None)` compares one time point at a time. The tensor `X_time` must have `(time, matched unit, feature)` shape; Y is `(matched unit, feature)`. Time samples must be finite and strictly increasing.

The optional `blocks` argument supplies one experiment-defined label per observation. Each permutation shuffles correspondence **within** each block only, rather than mixing observations across blocks. This is appropriate only when units really are exchangeable within the block: block labels alone do not justify inference. A separate maximum absolute RSA null distribution across time supports familywise-adjusted p-values.

A peak means representational similarity is strongest at that sampled time in these data. It is **not** a causal neural delay. fMRI measures delayed hemodynamic responses and needs separate interpretation.
