# Validation record — Neuro Multimodal Alignment 0.5.0

The checks below evaluate reproducibility and numerical behaviour. They do not establish correspondence between biological systems in real EEG, MEG or fMRI recordings.

## Local release checks

| Check | Recorded outcome | Scope |
| --- | --- | --- |
| Original archive | 34 passing tests | Before improvements |
| Revised implementation | 49 passing tests | Synthetic, numerical and input-validation checks |
| Statement coverage | 93% | Software coverage, not scientific validity |
| Scientific figures | Four PNG and four SVG generated | Seeded synthetic datasets |
| Python examples | Three executed | Synthetic data |
| Jupyter notebooks | First three executed | Synthetic data |
| Source compilation | Passed | Source, tests, examples and reproduction scripts |
| MNE adapter | Offline synthetic MNE test passed | No external recordings |

## Published repository

GitHub Actions has successfully run cross-platform tests, Python lint and package builds on published commits. See the [workflow history](https://github.com/lathushanrk-cmd/neuro-multimodal-alignment/actions) for the results associated with a specific revision. Remote runner success does not replace installation on an independent laboratory workstation.

## Scientific safeguards

- Observation identifiers are checked; the program cannot determine whether a researcher has *correctly* matched the underlying participants, stimuli or trials.
- CCA transformations are fitted on training data; held-out measurements are reported to limit training-score overfitting.
- Group-aware splits are available where repeated measurements must not cross folds.
- RSA is a descriptive geometry comparison and is not automatically cross-validated.
- Procrustes applies a mathematical rotation/reflection, which does not imply that two measurements share a biological origin.
- Temporal RSA uses user-declared exchangeability blocks and max-statistic corrections where specified. Peaks are not direct estimates of causal propagation or neural transmission times.
- Shared/private predictive analyses describe what a chosen linear model captures; they do not uniquely identify underlying biological information.

## Not independently established

- Complete execution of the MNE EEG/MEG example on downloaded recordings.
- Replication on independent participants, recording sessions or instruments.
- Biological or clinical validation, peer review or a software DOI.
- The scientific truth of any user-supplied pairing or independence assumptions.

## Reproduce software checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python reproduce/make_figures.py
python examples/01_ground_truth_simulation.py
python examples/02_rsa_demo.py
python examples/03_crossvalidated_cca.py
```

These results support the release as a computational research prototype. They should not be presented as empirical neuroscience discoveries.
