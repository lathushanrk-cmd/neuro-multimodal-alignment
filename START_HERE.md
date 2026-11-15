# Getting started with Neuro Multimodal Alignment

The main [README](README.md) explains the research question and links to each method. For correct use with your own measurements, read [data alignment](docs/data_alignment.md) before running a comparison.

Install in a Python 3.10+ environment from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/01_ground_truth_simulation.py
python examples/02_rsa_demo.py
python examples/03_crossvalidated_cca.py
```

Generate the synthetic figures with `python reproduce/make_figures.py`.

The [optional EEG/MEG example](examples/04_mne_eeg_meg_demo.py) requires external recordings and has not been verified as a complete empirical analysis. See [validation](VALIDATION_REPORT.md) for the checks actually performed.
