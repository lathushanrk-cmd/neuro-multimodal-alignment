# Getting started

## Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/01_ground_truth_simulation.py
python reproduce/make_figures.py
```

## Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/01_ground_truth_simulation.py
python reproduce/make_figures.py
```

All figures under `assets/` are generated with fixed seeds. The public EEG/MEG adapter is optional and requires a separate dataset download.
