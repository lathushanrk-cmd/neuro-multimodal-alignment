# START HERE — Neuro Multimodal Alignment

**This is Project 2, completely separate from Neural State Space Lab.** It studies correspondences between matched neural representations, rather than state perturbations and recovery.

Read [README](README.md), [matching requirements](docs/data_alignment.md) and [validation](VALIDATION_REPORT.md).

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python reproduce/make_figures.py
python examples/01_ground_truth_simulation.py
python examples/02_rsa_demo.py
python examples/03_crossvalidated_cca.py
```

All included scientific plots are simulated. Optional external sample EEG+MEG analysis is a separate unverified experiment, not an EEG-fMRI converter or a demonstrated real-data discovery.
