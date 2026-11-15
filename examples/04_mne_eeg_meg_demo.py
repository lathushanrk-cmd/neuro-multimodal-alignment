"""OPTIONAL REAL-DATA ADAPTER: Requires pre-downloaded MNE sample data.

Uses simultaneous EEG and MEG sensor signals from the SAME recorded epochs.
This demonstration illustrates a *single recording-session* comparison,
not population generalization. It has not been executed on downloaded data
in the current development environment.

Install: pip install -e '.[realdata]'
Fetch MNE sample from https://mne.tools/stable/overview/datasets_index.html
MNE sample dataset is relatively large; this is an optional example, not a prerequisite.
Use the local MNE_DATASETS_SAMPLE_PATH environment variable or data_path().
"""
import numpy as np
from nma.data import AlignedDataset
from nma.rsa import rsa_score

def extract_aligned_from_raw(raw):
    """Extract explicitly matched EEG/MEG epoch window features; no inference."""
    import mne
    events=mne.find_events(raw,stim_channel='STI 014',verbose='ERROR')
    epochs=mne.Epochs(raw,events,event_id={'auditory_left':1,'auditory_right':2,
                       'visual_left':3,'visual_right':4},tmin=0,tmax=.35,
                       baseline=None,preload=True,reject_by_annotation=True,verbose='ERROR')
    eeg=epochs.get_data(picks='eeg',copy=True)
    meg=epochs.get_data(picks='mag',copy=True)
    if eeg.shape[1]<2 or meg.shape[1]<2 or len(eeg)<10:
        raise RuntimeError('Insufficient data after epoching')
    sample_start=int(.1*epochs.info['sfreq'])
    eeg=np.mean(eeg[:,:,sample_start:],axis=2)
    meg=np.mean(meg[:,:,sample_start:],axis=2)
    return AlignedDataset(eeg,meg,range(len(eeg)),pairing='observation')

def run():
    import mne
    path=mne.datasets.sample.data_path(download=False)
    if not path or not (path/'MEG'/'sample'/'sample_audvis_raw.fif').exists():
        raise FileNotFoundError('MNE sample dataset not installed. See docs/open_data.md; no automatic large download.')
    raw=mne.io.read_raw_fif(path/'MEG'/'sample'/'sample_audvis_raw.fif',preload=False,verbose='ERROR')
    raw.load_data().filter(l_freq=1.,h_freq=30.,verbose='ERROR')
    dataset=extract_aligned_from_raw(raw)
    print('Single-session EEG/MEG descriptive RSA (Euclidean):',rsa_score(dataset,metric='euclidean'))
    print('Interpretation warning: sensor scaling and trial dependencies affect results. Not population inference.')

if __name__=='__main__': run()
