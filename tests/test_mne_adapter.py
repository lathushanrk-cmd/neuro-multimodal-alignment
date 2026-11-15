"""Optional MNE integration (synthetic RawArray, no real recordings)."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

def test_synthetic_raw_eeg_meg_events():
    mne=pytest.importorskip('mne')
    path=Path(__file__).parents[1]/'examples'/'04_mne_eeg_meg_demo.py'
    spec=importlib.util.spec_from_file_location('mne_demo',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sf=100.
    info=mne.create_info(['EEG 001','EEG 002','MEG 001','MEG 002','STI 014'],sf,
                         ch_types=['eeg','eeg','mag','mag','stim'])
    rng=np.random.default_rng(8)
    data=rng.normal(0,1e-12,(5,2400))
    data[0:2]*=1e5
    data[-1]=0
    for i,code in enumerate(([1,2,3,4]*3)):
        data[-1,100+i*170]=code
    raw=mne.io.RawArray(data,info,verbose='ERROR')
    aligned=module.extract_aligned_from_raw(raw)
    assert aligned.X.shape==(12,2)
    assert aligned.Y.shape==(12,2)
    assert aligned.pairing=='observation'
