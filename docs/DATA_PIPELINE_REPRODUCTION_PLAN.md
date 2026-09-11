# Data Pipeline Reproduction Plan

This is a plan only. It was not executed in Phase 2.

## Stage 1: Raw data verification

- **Input:** single-site CSVs and `backend/stratowatch_multi_site/data/raw/data/`.
- **Transformation:** verify filenames, site IDs, columns, datetimes, duplicates, gaps, missingness, and coordinates in a temporary workspace.
- **Expected output:** read-only inventory and source manifest.
- **Validation:** all expected site files and schemas match recorded baseline facts.
- **Risk:** source files may not be the files used to create frozen artifacts.

## Stage 2: Single-site preprocessing

- **Input:** site 1 train/unseen CSVs and a captured environment.
- **Transformation:** run the existing notebook logic in a temporary copy only.
- **Expected output:** temporary cleaned frames, residual scaler, feature metadata, and sequences.
- **Validation:** compare shapes, feature order, finite values, and residual reconstruction contract to frozen artifacts.
- **Risk:** current `y_res_scaler.pkl` is invalid and current feature metadata is 134 versus sequence width 179.

## Stage 3: Multi-site alignment

- **Input:** seven train and seven unseen site CSVs.
- **Transformation:** execute `03_align_multisite.py` against copied inputs and a temporary output directory.
- **Expected output:** aligned NPZ files and metadata.
- **Validation:** compare site order, union timeline lengths, columns, masks, and zero-imputation behavior.
- **Risk:** the script writes fixed relative output paths unless isolated before execution.

## Stage 4: Window generation

- **Input:** temporary `train_aligned.npz`.
- **Transformation:** execute the existing `04_make_windows.py` logic in an isolated copy.
- **Expected output:** 24-hour/6-hour/stride-1 window NPZ.
- **Validation:** check tensor layout, masks, feature order, target offsets, and overlap.
- **Risk:** large memory use and production-path writes if isolation is incomplete.

## Stage 5: Chronological splitting

- **Input:** temporary window NPZ.
- **Transformation:** execute the existing `05_split_windows.py` logic in isolation.
- **Expected output:** 70/15/15 chronological partitions.
- **Validation:** check boundary indices and document shared context around boundaries.
- **Risk:** overlapping stride-one windows create a known leakage concern; do not change it during reproduction.

## Stage 6: Normalization and feature engineering

- **Input:** temporary split artifacts and coordinate file.
- **Transformation:** execute notebook cells in a temporary project copy after recording the original cell sources and paths.
- **Expected output:** normalized intermediate, 28-feature final NPZ, final feature scaler, and target scaler.
- **Validation:** compare feature order, selected scaled fields, mask behavior, train-only statistics, and target metadata.
- **Risk:** notebook contains stale absolute paths and writes adjacency/final artifacts.

## Stage 7: Artifact validation

- **Input:** temporary artifacts.
- **Transformation:** load-only validation.
- **Expected output:** temporary integrity report.
- **Validation:** compare shapes, keys, metadata, scaler dimensions, adjacency shape, and coordinate order.
- **Risk:** numerical equivalence may fail even when schemas match.

## Stage 8: Baseline comparison

- **Input:** frozen artifacts and temporary regenerated artifacts.
- **Transformation:** no training; compare metadata and, only after explicit approval, run evaluation in an isolated environment.
- **Expected output:** comparison report, not replacement artifacts.
- **Validation:** no production artifact modified; numerical differences are explained or escalated.
- **Risk:** evaluator execution can write plots/metrics and uses shared paths unless isolated.
