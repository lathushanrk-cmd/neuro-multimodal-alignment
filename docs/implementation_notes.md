# Design decisions and interpretation

This document describes choices that can be verified in the current source. It is not a reconstructed personal development diary or a claim about who wrote each function.

## What is being matched?

Two matrices may describe the same stimuli, conditions, trials or participants in different feature spaces. The `data` module stores the declared matching relationship and validates identifiers. Equal row counts cannot establish scientifically correct pairing. A researcher must confirm that the identifiers mean the same thing in the underlying experiment.

## Why multiple alignment methods?

- **RSA** compares dissimilarity among items and does not require the two matrices to have the same number of features. The implemented basic score is descriptive, not held-out RSA.
- **CCA** learns linear projections with high cross-dataset correlation. Flexible projections can fit noise, so held-out evaluation, dimensional safeguards and training-only transformations are important.
- **Procrustes** studies whether representations are related by an orthogonal coordinate transformation. An orthogonal fit is a geometric description, not proof of shared biological sources.
- **Shared/private prediction** estimates how much of one dataset can be predicted from the other under a chosen model. The estimate is model-dependent.
- **Temporal RSA** repeats correspondence measurements through time. Apparent peaks must be assessed against valid nulls and cannot automatically be interpreted as causality or transmission delay.

The methods answer related but different questions. They should not be pooled into a single number called 'neural alignment'.

## Why are synthetic controls included?

A generator with specified shared factors provides a known-answer setting. Independently generated matrices provide a null setting. A high-dimensional CCA demonstration shows that strong training correlations can occur without generalizable shared structure.

These experiments check numerical and statistical behaviour for constructed data. They are not empirical results from EEG, MEG or fMRI.

## Why do groups and permutation blocks matter?

Repeated measurements from a single participant or session are dependent. Cross-validation must avoid leakage across independent evaluation units, and permutation tests require that the shuffled units are exchangeable under the null hypothesis. The implementation offers options for grouping and blocking; it cannot decide the right experimental design without researcher-supplied metadata.

## What remains unverified?

- Execution of the full optional MNE EEG/MEG script on downloaded sample data.
- Whether matching identifiers in a future scientific dataset are correct.
- Independent participant/session generalization.
- Biological significance of representation similarity.

Refer to [source](../src/nma/), [tests](../tests/), [validation](../VALIDATION_REPORT.md), and the [GitHub Actions history](https://github.com/lathushanrk-cmd/neuro-multimodal-alignment/actions) for the testable implementation and outcomes.
