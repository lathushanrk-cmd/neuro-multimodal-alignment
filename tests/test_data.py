import numpy as np
import pytest
from nma.data import AlignedDataset

def test_valid(aligned):
    assert aligned.X.shape == (100,24)
    assert aligned.pairing == 'condition'
    assert len(aligned.subset([0,1,2]).sample_ids)==3

def test_bad_dimensions():
    with pytest.raises(ValueError): AlignedDataset(np.ones((4,2)),np.ones((3,3)),range(4),'condition')

def test_bad_ids():
    with pytest.raises(ValueError): AlignedDataset(np.ones((4,2)),np.ones((4,3)),[1,1,2,3],'condition')
    with pytest.raises(ValueError): AlignedDataset(np.ones((4,2)),np.ones((4,3)),[1,2,3,None],'condition')

def test_nonfinite():
    x=np.ones((4,2)); x[0,0]=np.nan
    with pytest.raises(ValueError):AlignedDataset(x,np.ones((4,2)),list(range(4)),'condition')

def test_explicit_pairing():
    with pytest.raises(ValueError):AlignedDataset(np.ones((4,2)),np.ones((4,2)),range(4),'unpaired')

def test_reorder_ids():
    a=AlignedDataset.from_mismatched_rows(np.eye(4),np.arange(4)[:,None],[0,1,2,3],[2,0,3,1],'condition')
    assert a.Y[:,0].tolist()==[1,3,0,2]

def test_pairing_extra_rows():
    with pytest.raises(ValueError):AlignedDataset.from_mismatched_rows(np.ones((4,2)),np.ones((4,2)),[0,1,2,3],[0,1,2,8],'condition')
