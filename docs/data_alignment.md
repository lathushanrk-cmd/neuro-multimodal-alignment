# Match the correct observations

Provide finite 2-D feature matrices X and Y, explicitly declared `pairing` and matching unique `sample_ids`. Valid pairing types are `observation`, `condition`, and `subject`. They indicate the intended matching unit, not a guarantee of correct experiment design.

Use `AlignedDataset.from_mismatched_rows(X, Y, x_ids, y_ids, pairing=...)` to reorder Y according to independently supplied identifiers. Duplicate, missing, extra and blank IDs are rejected. Metadata must have one row per observation and no missing entries.

**Two unrelated matrices cannot be treated as paired simply because their row counts match.** Truly unpaired measurements require a separately justified scientific strategy outside the direct paired-analysis API. For repeated participants or sessions, pass research-defined group labels to methods that support group-aware cross-validation.
