# Explain and demonstrate Neuro Multimodal Alignment

These notes support a reproducible demonstration; they are not a statement that a formal interview has already been completed.

## Start the demonstration

From the root of the checked-out repository:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/01_ground_truth_simulation.py
python examples/02_rsa_demo.py
python examples/03_crossvalidated_cca.py
```

The first three examples are synthetic and require no participant recordings.

## Explain the idea in two minutes

EEG, MEG and fMRI do not measure exactly the same physical quantity. Instead of comparing raw sensor values, this toolkit studies whether matched experimental items preserve related patterns of similarity and difference in different sets of features.

The researcher must identify valid corresponding observations. The toolkit can then apply RSA to compare pairwise dissimilarities, CCA to find held-out correlated linear combinations, and Procrustes to examine correspondence after an orthogonal coordinate transformation. Optional temporal RSA examines how correspondence changes across sampled times.

The result is evidence about the chosen **mathematical relationship between features**, not proof that measurement modalities contain biologically identical codes.

## Guided file tour

1. Open `src/nma/data.py`. Explain why explicit identifiers matter and why matching rows by position alone is unsafe.
2. Open `src/nma/rsa.py`. Show how RDMs are formed and compared.
3. Open `src/nma/cca.py`. Identify the training/held-out separation, scaling and any dimension safeguards.
4. Open `src/nma/procrustes.py`. Explain an orthogonal transformation and what a held-out error measures.
5. Show `assets/cca_leakage.png` to illustrate how training correlation can overstate generalization.
6. Show `assets/temporal_alignment.png` and explain the distinction between an injected simulated lag and an empirically measured physiological delay.

## Questions to prepare for

**Why not compare EEG and fMRI measurements directly?**  
Their sensors, units, temporal responses and spatial mixing differ. Representational comparisons ask a narrower, experimentally controlled question.

**Why is an ID check not enough?**  
An identifier can be unique and correctly formatted but still name the wrong stimulus, trial or participant. Only the study design and raw metadata can establish the true correspondence.

**What does RSA tell you?**  
Whether the pattern of pairwise relationships among matched items is similar across representations. A high score does not prove identical neural mechanisms.

**What can go wrong with CCA?**  
High-dimensional models can find strong in-sample correlations in noise. Held-out evaluation and appropriate independent groups are essential.

**How do permutation tests work?**  
They compare the observed statistic to values under deliberately broken correspondences, but only when the shuffled units satisfy a valid exchangeability argument.

**Has the toolkit shown actual EEG/MEG alignment?**  
Not through a verified downloaded recording in the current release. The package supplies an optional adapter, while the published figures are synthetic.

**What should happen next?**  
Run the MNE example on correctly obtained sample data, confirm events and recording metadata, compare with suitable confound controls, and avoid across-subject claims from a single session.
