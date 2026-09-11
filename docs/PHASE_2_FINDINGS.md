# Phase 2 Findings

## 1. Executive Summary

The current data pipeline is only partially reproducible. Multi-site alignment, masking, windowing, and chronological splitting are implemented in scripts and are deterministic from their inputs. The active 28-feature and target-scaled artifacts are produced through notebook-only stages with stale absolute paths and no hash-linked provenance. The single-site pipeline cannot be reproduced end-to-end from the present artifacts because feature metadata does not match sequence width.

No model, preprocessing algorithm, data artifact, scaler, checkpoint, graph, API, frontend, or generated output was changed.

## 2. Single-Site Data Lineage

`preprocessing_clean.ipynb` reads site 1 train/unseen CSVs, creates hourly datetime indexes, sorts/reindexes, adds missingness flags, replaces infinities, interpolates and fills values, engineers calendar/wind/lag/rolling features, constructs residual targets, scales inputs/residuals, creates 24-step sequences, and saves `sequences.npz`, `feature_list.json`, and `y_res_scaler.pkl`.

`train.py` and the baseline scripts split the saved sequences chronologically 70/15/15. `evaluate.py` loads the transformer and residual scaler, inverse-scales residual predictions/truths, and reconstructs `forecast + residual`.

## 3. Single-Site Feature Schema

The observed raw train schema has 16 columns and the sequence artifact has width 179. The notebook source includes raw fields, missing flags, cyclic fields, wind features, six lag horizons, and rolling mean/std features. The persisted `feature_list.json` has only 134 names. The exact run or artifact-producing code that caused this mismatch is UNKNOWN; the current notebook cell would ordinarily save the current dataframe column list alongside the sequence width.

The repository README also claims outlier clipping, but the inspected notebook source does not establish a quantile-clipping step. This is a documentation/code discrepancy, not a Phase 2 fix.

## 4. Single-Site Scaler Provenance

The notebook creates a two-output `sklearn.preprocessing.StandardScaler` over residual O3/NO2 values and serializes it with `joblib.dump` to `data/y_res_scaler.pkl`. The artifact loads successfully with `joblib.load()` and emits an `InconsistentVersionWarning` because it was serialized under scikit-learn 1.8.0 and is currently loaded under 1.7.2. Its fitted parameters match the scratch reproduction within floating-point tolerance.

## 5. Multi-Site Raw Data

There are seven train and seven unseen site files. Train files contain 16 columns; unseen files contain the 14 non-target columns. All inspected files have parseable datetimes, zero duplicate timestamps, and no timezone field. Raw date values span 2019-07-10 through 2024-06-30. Aggregate missingness is 18.1252% for train and 20.6917% for unseen, concentrated in satellite fields. Raw sampling includes one-hour gaps and longer gaps, so source rows are not continuous hourly series.

## 6. Multi-Site Alignment

`03_align_multisite.py` extracts numeric IDs from filenames and sorts train site IDs. It forms separate train and unseen union timelines, sorts them, reindexes each site, retains all non-target columns, zero-imputes NaNs, and writes X/Y masks based on pre-imputation validity. Site coordinates are separate and ordered by site ID. The code does not enforce raw regularity; it creates the common hourly grid through reindexing.

## 7. Multi-Site Feature Schema

The active 28 features are the 14 raw non-target fields, six cyclical calendar fields, six satellite observation/age fields, and two broadcast coordinate fields. The exact index-level schema and scaling status are documented in `docs/MULTISITE_FEATURE_SCHEMA.md`.

## 8. 14-vs-28 Investigation

The 14-feature `global_scaler.json` belongs to the first normalization stage over the raw aligned feature set. The notebook then appends six time encodings, six satellite missingness/age features, and two coordinates. Cell 15 applies a partial train-only scaler to five continuous engineered fields and writes the 28-feature final artifact. The active NPZ and ST checkpoint both confirm 28. The 14-feature scaler is not compatible with the active final model input and is not loaded by active runtime evaluation code.

## 9. Window Generation

`04_make_windows.py` uses `Tin=24`, `Tout=6`, and `STRIDE=1`. It maps `X[t:t+24]` to `Y[t+24:t+30]` and carries corresponding masks. Windows overlap by 29 of their combined 30 time positions when adjacent, and input windows overlap by 23 positions.

## 10. Train/Validation/Test Split

`05_split_windows.py` uses `int(N*0.70)` train windows, `int(N*0.15)` validation windows, and the remainder as test. It performs direct chronological slicing, not random sampling. The final active artifact contains 21,182 train, 4,539 validation, and 4,539 test windows.

## 11. Leakage Analysis

The split is chronological but has no purge gap. Since stride-one windows overlap, the last train window and first validation window share input history; the same applies at validation/test boundaries. This is a potential temporal leakage concern. It is preserved as a baseline finding and was not removed.

Single-site scaling also occurs before the saved sequence split, so its input/residual scaler fitting population is not strictly train/validation/test isolated according to the notebook source.

## 12. Normalization

Normalization has four stages: mask-aware global scaling of the original 14 X features, appending time features, appending satellite observation/age and coordinates, partial train-only scaling of five engineered continuous features, and train-only target scaling. Masks are preserved; invalid raw values are zero-filled before scaling and excluded from scaler statistics. Details and UNKNOWN items are in `docs/NORMALIZATION_AUDIT.md`.

## 13. Artifact Integrity

The final multi-site NPZ, JSON scalers, adjacency, coordinates, and two multi-site checkpoints are present and loadable. The single-site sequence, transformer checkpoint, and Joblib residual scaler load. Feature metadata remains width-inconsistent. Full status is in `docs/ARTIFACT_INTEGRITY_REPORT.md`.

## 14. Reproducibility Assessment

Multi-site raw-to-final preprocessing is **PARTIALLY REPRODUCIBLE**. Single-site raw-to-evaluation reproduction is **NOT REPRODUCIBLE** from the current artifact set. Frozen-artifact multi-site model loading is **PARTIALLY REPRODUCIBLE**. The detailed classification is in `docs/REPRODUCIBILITY_DATA_PIPELINE.md`.

## 15. Confirmed Problems

- Single-site `feature_list.json` has 134 names for a 179-wide sequence.
- Single-site residual scaler requires Joblib loading and a compatible scikit-learn version.
- Multi-site final processing is notebook-dependent.
- Final normalization cell contains a stale absolute path.
- Global 14-feature scaler and final 28-feature schema coexist without a provenance manifest.
- Stride-one split boundaries share temporal context.
- Raw inputs and final artifacts are not hash-linked.

## 16. Unknowns

- Exact run that produced the current single-site metadata/scaler artifacts.
- Whether current raw CSVs are byte-for-byte the inputs used for the frozen final NPZ.
- Exact raw timezone semantics.
- Exact final NPZ numerical equivalence from a temporary reproduction.
- Complete original Python package versions and random states.

## 17. Deferred Fixes

No fixes were made to feature counts, scaler files, notebook paths, missing-value handling, scaling, leakage, feature ordering, split behavior, or artifact provenance. These require a separate approved change phase.

## 18. Recommended Phase 3

Phase 3 should establish a temporary, isolated reproduction harness that copies raw inputs and notebook/scripts into a scratch directory, records environment and artifact hashes, and compares generated schemas/statistics against the frozen artifacts without overwriting them. Only after that comparison should any repair of metadata, scaler provenance, notebook paths, or leakage policy be considered.
