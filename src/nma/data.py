"""ID-aware paired observations. Pairing is always explicit."""
from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass(frozen=True)
class AlignedDataset:
    X: np.ndarray
    Y: np.ndarray
    sample_ids: object
    pairing: Literal["observation", "condition", "subject"]
    x_feature_names: object = None
    y_feature_names: object = None
    metadata: object = None

    def __post_init__(self):
        x, y = np.asarray(self.X, dtype=float), np.asarray(self.Y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("X and Y must be 2D matrices with the same number of matched rows")
        if x.shape[0] < 3 or min(x.shape[1], y.shape[1]) < 1:
            raise ValueError("Need at least three matched units and one feature per modality")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite features are unsupported: impute/clean before alignment")
        if self.pairing not in ("observation", "condition", "subject"):
            raise ValueError("Explicit pairing='observation', 'condition', or 'subject' required")
        if self.sample_ids is None:
            raise ValueError("Explicit independent sample IDs are required")
        ids = list(self.sample_ids)
        if len(ids) != x.shape[0] or len(set(map(str,ids))) != len(ids):
            raise ValueError("sample_ids must be unique, nonmissing, and match the number of rows")
        if any(v is None or str(v).strip() == '' or str(v).lower() == 'nan' for v in ids):
            raise ValueError("Missing or blank sample IDs")
        for names, size in ((self.x_feature_names, x.shape[1]), (self.y_feature_names, y.shape[1])):
            if names is not None and len(names) != size:
                raise ValueError("Feature names must align to matrix columns")
        if self.metadata is not None:
            if len(self.metadata) != x.shape[0]:
                raise ValueError("Metadata row count does not match matrices")
            if hasattr(self.metadata, 'isna') and bool(self.metadata.isna().any().any()):
                raise ValueError("Metadata contains missing values; clean or explicitly document first")
        object.__setattr__(self, 'X', x.copy())
        object.__setattr__(self, 'Y', y.copy())
        object.__setattr__(self, 'sample_ids', tuple(ids))

    def subset(self, index):
        index = np.asarray(index)
        return AlignedDataset(self.X[index], self.Y[index],
            np.asarray(self.sample_ids, dtype=object)[index].tolist(), self.pairing,
            self.x_feature_names, self.y_feature_names,
            None if self.metadata is None else self.metadata.iloc[index].reset_index(drop=True) if hasattr(self.metadata, 'iloc') else np.asarray(self.metadata)[index])

    @classmethod
    def from_mismatched_rows(cls, X, Y, x_ids, y_ids, pairing, **kwargs):
        """Reorder Y by independently supplied IDs. Fail on missing/extras/duplicates."""
        xi, yi = list(x_ids), list(y_ids)
        if any(v is None or str(v).strip() == '' or str(v).lower() == 'nan' for v in xi+yi):
            raise ValueError('Identifiers must not be missing or blank')
        if len(set(map(str,xi))) != len(xi) or len(set(map(str,yi))) != len(yi):
            raise ValueError('Duplicate identifiers')
        if set(xi) != set(yi):
            raise ValueError('Both modalities must contain exactly the same IDs')
        pos = {v: i for i,v in enumerate(yi)}
        return cls(X, np.asarray(Y)[[pos[v] for v in xi]], xi, pairing, **kwargs)
