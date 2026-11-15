# GitHub Project #2 Master Guide — Neuro Multimodal Alignment

**Repository:** `neuro-multimodal-alignment`  
**Project type:** Open-source multimodal neuroscience analysis toolkit  
**Primary language:** Python  
**Portfolio goal:** demonstrate multimodal neuroscience, representation analysis, temporal alignment, statistics, software engineering, reproducibility, and scientific restraint.

---

# 1. Locked project identity

# **Neuro Multimodal Alignment**
### A Reproducible Toolkit for Comparing Shared Neural Representations Across Measurement Modalities

## One-sentence description

> A Python toolkit for quantifying shared and modality-specific representational structure across neural measurements such as EEG/MEG and fMRI using explicit alignment, lag, cross-validation, and permutation frameworks.

## Core question

> **When two neuroscience modalities observe related brain processes, which representational structures are genuinely shared, and which are modality-specific?**

## Not this

- not “convert EEG into fMRI”;
- not a mind-reading system;
- not a generic deep-learning predictor;
- not proof that aligned latent dimensions have identical biological meaning;
- not a tool that assumes simultaneously recorded data when data are unpaired;
- not a conference-analysis repository.

---

# 2. Scientific motivation

Different modalities measure different transformations of neural activity:

- EEG/MEG: high temporal resolution, indirect population electromagnetic activity;
- fMRI: slower hemodynamic signal, high spatial coverage;
- calcium imaging/spiking: different temporal/biophysical properties again.

A useful open-source tool should make the assumptions of cross-modal comparison explicit rather than hiding them inside one model.

Core workflow:

> **modality A representation → modality B representation → harmonized experimental units → alignment model → cross-validation → shared/private structure → temporal-lag analysis → null tests → interpretable figures**

---

# 3. Version 1.0 scope

Support two generic feature matrices:

\[
X \in \mathbb{R}^{n\times p}, \quad Y\in\mathbb{R}^{n\times q}
\]

where rows represent matched experimental units such as conditions, stimuli, trials aggregated appropriately, or subjects.

V1 must include:

1. input validation;
2. z-scoring/normalization utilities;
3. PCA pre-reduction;
4. Canonical Correlation Analysis (CCA);
5. orthogonal Procrustes alignment;
6. Representational Similarity Analysis (RSA);
7. shared-vs-private variance summary;
8. temporal lag scanning for time-resolved modality A against modality B representation;
9. cross-validation;
10. permutation inference;
11. simulated ground-truth benchmark;
12. one open-data demonstration.

Deep neural networks are deliberately excluded from V1.

---

# 4. Fundamental distinction: paired vs unpaired data

This repository must teach users that “multimodal” does not automatically mean the same people were measured simultaneously.

## Paired

Same subjects/trials/conditions have corresponding observations.

## Condition-paired

Different recordings but shared stimuli/conditions allow comparison at condition-representation level.

## Unpaired

No direct observation-level correspondence. Only methods with defensible shared anchors should be used.

The API must require users to declare the alignment unit.

Never silently pair rows by array order.

---

# 5. Method 1 — Representational Similarity Analysis

For each modality, compute a representational dissimilarity matrix (RDM):

\[
R_X(i,j)=d(X_i,X_j)
\]

\[
R_Y(i,j)=d(Y_i,Y_j)
\]

Then compare upper triangles, e.g. Spearman correlation:

\[
\rho_{RSA}=corr(rank(vec(R_X)),rank(vec(R_Y)))
\]

This asks whether relational geometry is shared even when raw feature spaces differ.

Primary distances:

- correlation distance;
- Euclidean distance after appropriate standardization.

Do not provide dozens of distances initially.

---

# 6. Method 2 — CCA

CCA seeks projections:

\[
u=Xa, \quad v=Yb
\]

maximizing:

\[
corr(u,v)
\]

Critical rule: report **cross-validated** canonical correlations. In-sample CCA can badly overfit high-dimensional data.

Use PCA or regularization when dimensions are large relative to observations.

---

# 7. Method 3 — Orthogonal Procrustes

After dimensionality reduction to compatible dimensions, solve:

\[
R^*=\arg\min_{R^TR=I} ||XR-Y||_F
\]

Report held-out alignment error.

Do not interpret low Procrustes error as proof of common biological mechanism.

---

# 8. Temporal-lag alignment

For time-resolved electrophysiology representation \(X(t)\) and slower modality representation \(Y\):

\[
S(t)=Alignment(X(t),Y)
\]

Plot alignment across time.

If comparing electrophysiology with fMRI, optionally support a transparent hemodynamic forward-model module later, but do not force a single HRF assumption into the core package.

Any lag search must correct for multiple comparisons or use cluster/permutation logic appropriate to the design.

---

# 9. Shared vs private structure

Define carefully.

“Shared” should mean variance/geometry predictable across modalities under the chosen model and held-out evaluation.

“Private” means residual modality-specific structure under that model.

Do not claim it means biologically unique information without further evidence.

Possible V1 summaries:

- held-out CCA correlation;
- held-out Procrustes reconstruction score;
- RSA correlation;
- variance explained by shared latent factors;
- residual variance.

---

# 10. Ground-truth simulation suite

This is essential.

Generate latent process:

\[
s \sim \mathcal{N}(0,\Sigma)
\]

Modality observations:

\[
X=sA+uB+\epsilon_X
\]

\[
Y=sC+vD+\epsilon_Y
\]

where:

- \(s\) = truly shared factors;
- \(u\) = X-private factors;
- \(v\) = Y-private factors;
- noise terms vary independently.

Simulations:

1. 100% shared;
2. 50% shared;
3. zero shared;
4. nonlinear mismatch;
5. high-dimensional low-n regime;
6. temporal lag;
7. row-misalignment failure;
8. confounded condition labels.

The software should recover known qualitative behavior.

---

# 11. Statistical validation

## Permutation null

Shuffle correspondence between matched units and recompute alignment.

## Cross-validation

Fit transformations on training observations and evaluate on held-out observations.

## Bootstrap

Resample independent units to estimate uncertainty where scientifically valid.

## Confound check

Allow nuisance regression or stratified permutations when metadata support it.

Never permute observations in a way that destroys subject/block dependencies incorrectly.

---

# 12. Data leakage rules

All learned transformations must be fit on training data only:

- scaler;
- PCA;
- CCA;
- feature selection;
- alignment transform.

Then transform test data.

This is one of the strongest ways to distinguish the project from a visually impressive but statistically invalid notebook.

---

# 13. Architecture

```text
neuro-multimodal-alignment/
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── pyproject.toml
├── .gitignore
├── .github/
│   └── workflows/
│       ├── tests.yml
│       └── lint.yml
├── docs/
│   ├── index.md
│   ├── data_alignment.md
│   ├── rsa.md
│   ├── cca.md
│   ├── procrustes.md
│   ├── temporal_alignment.md
│   ├── statistics.md
│   └── limitations.md
├── examples/
│   ├── 01_ground_truth_simulation.ipynb
│   ├── 02_rsa_demo.ipynb
│   ├── 03_crossvalidated_cca.ipynb
│   └── 04_open_neuro_demo.ipynb
├── src/
│   └── nma/
│       ├── __init__.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── rsa.py
│       ├── cca.py
│       ├── procrustes.py
│       ├── temporal.py
│       ├── shared_private.py
│       ├── statistics.py
│       ├── simulation.py
│       └── plotting.py
└── tests/
    ├── test_data.py
    ├── test_rsa.py
    ├── test_cca.py
    ├── test_procrustes.py
    ├── test_temporal.py
    └── test_simulation.py
```

---

# 14. Data object

Create a clear container such as:

```python
AlignedDataset(
    X,
    Y,
    sample_ids,
    x_feature_names=None,
    y_feature_names=None,
    metadata=None,
    pairing="condition"
)
```

Validation must catch:

- mismatched sample IDs;
- duplicates;
- missing rows;
- NaNs;
- incompatible dimensions;
- invalid pairing declarations.

---

# 15. Example API

```python
from nma import AlignedDataset
from nma.rsa import rsa_score
from nma.statistics import permutation_test

aligned = AlignedDataset(X, Y, sample_ids=stimulus_ids, pairing="condition")
score = rsa_score(aligned, metric="correlation")
null = permutation_test(aligned, method="rsa", n_permutations=5000, random_state=42)

print(score)
print(null.pvalue)
```

For CCA:

```python
from nma.cca import CrossValidatedCCA

model = CrossValidatedCCA(n_components=3, n_splits=5, random_state=42)
result = model.fit_evaluate(aligned)
```

---

# 16. Open-data demonstration

Choose one openly licensed/public neuroscience dataset where two modalities or condition-level representations can be compared defensibly.

Before locking it, verify:

- access;
- license;
- modality pairing;
- sample/condition identifiers;
- manageable download size;
- preprocessing requirements;
- citation requirements.

Do not make the repository depend on a 500 GB download for its first demo.

A synthetic demo must always remain available so installation can be tested without external data.

---

# 17. Figures

## Hero Figure

Four panels:

1. Modality A feature space;
2. Modality B feature space;
3. shared latent/representational alignment;
4. held-out alignment score against permutation null.

## Figure 2 — Ground-truth recovery

x-axis: true shared fraction.  
y-axis: estimated alignment/shared score.

## Figure 3 — Leakage demonstration

Compare invalid in-sample alignment against correct held-out alignment in high-dimensional simulation.

This is educational and shows methodological maturity.

## Figure 4 — Temporal alignment

Time-resolved alignment curve with null/uncertainty band.

---

# 18. Tests

Known-answer tests:

- identical RDMs → RSA near 1;
- randomized independent representations → expected near-null behavior;
- known orthogonal rotation → Procrustes recovers rotation within tolerance;
- synthetic shared latent factors → CCA recovers cross-modal association;
- shuffled pairing → reduced alignment;
- preprocessing fitted only on training folds;
- temporal lag simulation → peak recovered near true lag.

Also test malformed inputs and deterministic random states.

---

# 19. Git/GitHub beginner workflow

## First-time sequence

1. Create GitHub account.
2. Install Git.
3. Create public repository `neuro-multimodal-alignment`.
4. Clone it locally.
5. Open the folder in Visual Studio/your Python environment.
6. Create virtual environment.
7. Add `pyproject.toml`.
8. Add `src/nma/`.
9. Add one tiny test.
10. Commit.
11. Push.
12. Confirm files appear online.

Commands:

```bash
git clone <url>
cd neuro-multimodal-alignment
git checkout -b feature/rsa
git add .
git commit -m "Implement cross-validated RSA utilities"
git push -u origin feature/rsa
```

Open a Pull Request, inspect CI, merge when green.

---

# 20. Commit history standard

Your history should tell the engineering story.

Good sequence:

```text
Initialize package and CI
Add validated aligned-data container
Implement RDM construction
Add RSA permutation inference
Implement cross-validated CCA
Add Procrustes held-out evaluation
Add shared/private simulation suite
Add temporal lag analysis
Add open-data tutorial
Prepare v1.0 documentation
```

Avoid uploading a finished 5,000-line project in one commit. That hides the development process.

---

# 21. README

Top of README:

# Neuro Multimodal Alignment

> Cross-validated tools for testing shared representational structure across neuroscience modalities.

Then:

- CI/license/DOI badges;
- hero figure;
- Why multimodal alignment is difficult;
- supported methods;
- installation;
- 60-second synthetic example;
- data assumptions;
- cross-validation warning;
- tutorials;
- API docs;
- limitations;
- citation;
- license.

---

# 22. Documentation pages that matter

## `data_alignment.md`
Explain exactly what “paired”, “condition-paired”, and “unpaired” mean.

## `statistics.md`
Explain independent units, exchangeability, permutations, cross-validation.

## `limitations.md`
Explain that mathematical alignment does not imply identical neurobiology.

## `temporal_alignment.md`
Explain lag search, temporal multiple comparisons, hemodynamic limitations.

These pages are part of the scientific contribution.

---

# 23. Software quality gates

### Q0 — install
Fresh environment works.

### Q1 — input integrity
Bad pairing cannot silently pass.

### Q2 — known-answer simulations
Methods recover ground truth.

### Q3 — leakage
All fitted transforms remain fold-specific.

### Q4 — null calibration
Independent simulated modalities do not systematically produce significance.

### Q5 — documentation
Beginner can run synthetic tutorial.

### Q6 — CI
Tests run on push/PR.

### Q7 — release
Tagged release reproducible.

---

# 24. Scientific claims allowed

Allowed:

> “The toolkit quantifies cross-modal representational correspondence under specified alignment assumptions.”

> “Cross-validation and permutation tests are used to assess generalization and null expectations.”

Not allowed:

> “This tool identifies the true neural code shared by EEG and fMRI.”

> “Aligned dimensions are the same biological process.”

> “The package converts one modality into another.”

---

# 25. Performance benchmarking

Benchmark:

- RDM computation vs observations;
- CCA vs feature dimension;
- permutation runtime;
- temporal alignment runtime;
- memory usage.

Use small/medium/large synthetic configurations.

Document the machine specifications used.

---

# 26. Package/release standards

Use:

- `pyproject.toml`;
- `src/` layout;
- semantic versioning;
- `pytest`;
- GitHub Actions;
- `CITATION.cff`;
- CHANGELOG;
- tagged releases.

Optional after stability:

- PyPI package;
- Zenodo DOI;
- documentation site.

Do not rush PyPI before the API is stable.

---

# 27. Development roadmap

## Phase 0 — Git practice
Create throwaway practice repository and learn clone/add/commit/push/branch/PR.

## Phase 1 — Package skeleton
Installable package + CI + first test.

## Phase 2 — Data integrity
AlignedDataset + validation + tests.

## Phase 3 — RSA
RDMs + score + permutation.

## Phase 4 — Procrustes
Fit/evaluate + known rotation simulation.

## Phase 5 — CCA
Cross-validated CCA + dimensionality safeguards.

## Phase 6 — Simulation framework
Shared/private factors + leakage demo.

## Phase 7 — Temporal module
Time-resolved alignment + lag simulation.

## Phase 8 — Open-data example
One defensible public demonstration.

## Phase 9 — Documentation
Scientific assumptions + tutorials + hero figure.

## Phase 10 — v1.0
Tests green, release notes, optional DOI/PyPI.

---

# 28. Stop conditions / anti-bloat rules

Do not add a method if:

- it duplicates an existing method without clear value;
- you cannot validate it synthetically;
- its assumptions cannot be documented;
- it creates an impressive screenshot but no scientific utility;
- it requires huge compute with no benefit to the core question.

Do not turn V1 into a deep-learning library.

---

# 29. Final checklist

- [ ] clear paired/unpaired semantics
- [ ] synthetic ground truth
- [ ] RSA
- [ ] cross-validated CCA
- [ ] Procrustes
- [ ] temporal alignment
- [ ] permutation nulls
- [ ] no leakage
- [ ] tests
- [ ] CI
- [ ] documentation
- [ ] open-data demo
- [ ] hero figure
- [ ] reproducible scripts
- [ ] license
- [ ] citation file
- [ ] release
- [ ] limitations
- [ ] no restricted data

---

# 30. Final locked identity

> **Neuro Multimodal Alignment is an open-source Python toolkit for rigorously testing shared representational structure across neural measurement modalities using explicit pairing assumptions, cross-validation, ground-truth simulations, temporal alignment, and permutation inference.**

---

# 31. Authoritative implementation references

Use official/current documentation for:

- GitHub repositories, branches, pull requests and Actions;
- Python Packaging User Guide and `pyproject.toml`;
- scikit-learn model-selection and CCA APIs;
- MNE-Python and Nilearn where modality-specific handling is used;
- pytest;
- Zenodo if DOI archiving is enabled.
