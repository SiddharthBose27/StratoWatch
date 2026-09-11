# Phase 3 Reproduction Report

## 1. Objective

Determine which frozen StratoWatch data artifacts can be reproduced from the currently available raw inputs and source code, using an isolated scratch environment and without changing production artifacts.

## 2. Safety Guarantees

- Scratch root: `/tmp/stratowatch_phase3_reproduction/`.
- Raw inputs, notebooks, and scripts were copied before execution.
- Multi-site scripts wrote only under `/tmp/stratowatch_phase3_reproduction/multisite_project/`.
- Notebook path substitutions were limited to scratch `PROJECT_ROOT` values.
- No production NPZ, NPY, CSV, JSON scaler, PKL scaler, checkpoint, figure, notebook, or script was written.
- No model training, evaluation, frontend, or API execution was performed.

## 3. Frozen Artifact Manifest

The initial SHA-256 manifest is `/tmp/stratowatch_phase3_reproduction/baseline_manifest.json`. It records 37 source/input/artifact files with paths, sizes, timestamps, and hashes. The post-reproduction hash comparison reported `production_hash_changes []` for every recorded file.

## 4. Environment

| Item            | Installed value                  | Repository request/evidence              |
| --------------- | -------------------------------- | ---------------------------------------- |
| OS/architecture | macOS 15.7.3 arm64               | macOS workspace                          |
| Python          | 3.13.5                           | README recommends Python 3.12+           |
| pip             | 26.0.1                           | not pinned                               |
| NumPy           | 2.2.6                            | root unpinned; multi-site `>=1.24`       |
| pandas          | 2.3.3                            | root unpinned; multi-site `>=2.0`        |
| SciPy           | 1.16.3                           | single-site `>=1.10`                     |
| scikit-learn    | 1.7.2                            | root unpinned; multi-site `>=1.3`        |
| PyTorch         | 2.10.0                           | root unpinned; multi-site `>=2.1.0`      |
| joblib          | 1.5.2                            | single-site `>=1.3`                      |
| Jupyter         | notebook 7.5.2 / nbformat 5.10.4 | multi/single requirements `jupyter>=1.0` |

The production residual scaler emits an `InconsistentVersionWarning`: it was serialized with scikit-learn 1.8.0 and is being loaded with 1.7.2. The installed environment is evidence of the current reproduction attempt, not proof of the historical training environment.

## 5. Single-site Source Lineage

The executed source was `backend/stratowatch_single_site/preprocessing_clean.ipynb`, cells 0 through 12, with copied `site_1_train_data.csv` and `site_1_unseen_input_data.csv` placed at the notebook's expected relative paths. The notebook performs datetime creation, sorting, hourly reindexing, numeric missing flags, infinity replacement, time interpolation, forward/backward filling, cyclic/wind features, forecast lags, rolling statistics, residual construction, grouped feature scaling, residual scaling, and 24-step sequence creation.

## 6. Single-site Reproduction

The notebook completed successfully in scratch and generated:

- `X_seq=(43489,24,134)`
- `y_seq=(43489,2)`
- `feature_list.json` with 134 names
- a valid scratch `StandardScaler` serialized by joblib

The frozen artifact is `X_seq=(43489,24,179)`. The scratch feature list is byte-identical to the frozen 134-name `feature_list.json`. Target arrays match the frozen target array to approximately `1.8e-15` maximum absolute difference. The first 134 frozen X dimensions do not match scratch (`max_abs_diff` approximately `525.93`), so the frozen 179-wide sequence is not the current notebook output with 45 trailing features added.

**Status:** PARTIALLY REPRODUCED. Current notebook lineage reproduces the metadata and targets, but not the frozen X sequence.

## 7. 134 vs 179 Investigation

Evidence establishes:

1. The current notebook saves `feature_list = list(X_train_scaled.columns)`.
2. The current notebook produces 134 scaled X columns and 134-wide sequences.
3. The frozen `feature_list.json` is exactly the current notebook's 134-name list.
4. The frozen sequence contains 179 X features, including 45 dimensions with no corresponding persisted names.
5. The shared first 134 values differ substantially, so this is not merely an appended-feature mismatch.

The strongest evidence-based conclusion is **multiple artifact lineages or a historical preprocessing run**: current notebook + metadata produce 134, while the frozen sequence was produced by another feature/scaling state. The exact historical source of the 179 dimensions is **UNKNOWN**. Git history cannot resolve it: the repository has only the initial commit (`5e5ef4b`) for the relevant paths, and the ignored binary artifacts are not versioned.

## 8. Residual Scaler Investigation

The notebook generates the scaler with `joblib.dump(y_res_scaler, "data/y_res_scaler.pkl")`. The frozen scaler loads successfully with `joblib.load`, as does the scratch scaler. Their fitted parameters are equivalent:

- mean maximum absolute difference: approximately `7.1e-15`
- scale maximum absolute difference: `0.0`
- both are `StandardScaler` objects with two inputs

The frozen file fails with raw `pickle.load` (`invalid load key, '\\x09'` in the Phase 1 probe; another raw probe produced `\\x07`) because it is a joblib serialization, not a plain pickle stream suitable for direct loading. This refines the Phase 1 finding: the artifact is not proven corrupt; the intended loader works, with a scikit-learn version warning.

The scratch and frozen scaler bytes differ because serialization/environment metadata differs. Parameter equivalence is established; byte equivalence is not.

## 9. Multi-site Source Lineage

The copied scripts were run in order:

1. `01_unpack_and_inspect.py`
2. `02_check_all_sites.py`
3. `03_align_multisite.py`
4. `04_make_windows.py`
5. `05_split_windows.py`

The normalization notebook cells 1, 2, 3, 5, 6, 7, 15, and 16 were then executed with only the notebook `PROJECT_ROOT` redirected to scratch.

## 10. Alignment Reproduction

The copied alignment stage reproduced exactly:

- sites `[1,2,3,4,5,6,7]`
- train timeline length `30,289`
- unseen timeline length `13,008`
- 14 raw input features
- train and unseen aligned shapes
- masks and zero-imputation behavior

`train_aligned.npz` and `unseen_aligned.npz` are byte-for-byte identical to the frozen artifacts.

**Status:** EXACTLY REPRODUCED.

## 11. 28-feature Reproduction

The scratch notebook reproduced the sequence `14 -> 20 -> 26 -> 28`:

- 14 raw features
- six cyclical time features
- six satellite observation/age features
- two broadcast coordinate features

Feature names and order match the frozen final NPZ and `final_feature_scaler.json`. The final feature arrays are structurally equivalent, but their numeric values are not byte-identical because normalization statistics differ slightly.

**Status:** STRUCTURALLY EQUIVALENT, NUMERICALLY CLOSE, NOT EXACT.

## 12. 14 vs 28 Reproduction

The 14-feature representation is created by alignment and is input to the first notebook normalization stage. `global_scaler.json` is fitted over this representation. The notebook then appends 14 engineered features and fits `final_feature_scaler.json` only on three age channels and two coordinate channels. The target scaler is fitted after this stage.

The 14-feature scaler is therefore an intermediate artifact required by the current notebook lineage, not the active final model input. The active final NPZ/checkpoint use 28 features.

## 13. Normalization Reproduction

The scratch notebook reproduced all normalization stages and train-only mask-aware fitting logic. It did not reproduce frozen scaler bytes or exact arrays:

| Artifact                     | Result                  | Difference                                                            |
| ---------------------------- | ----------------------- | --------------------------------------------------------------------- |
| `global_scaler.json`         | structurally equivalent | mean max difference `1.14e-05`; std max difference `1.53e-05`         |
| `splits_normalized...npz`    | structurally equivalent | X max difference `0.0029296875`; masks/Y exact                        |
| `final_feature_scaler.json`  | structurally equivalent | mean max difference `0.00030517578125`; std max difference `1.91e-06` |
| `target_scaler.json`         | structurally equivalent | mean max difference `5.72e-06`; std max difference `1.91e-06`         |
| `splits_final...npz`         | structurally equivalent | X max difference `0.004015088`; Y unscaled exact                      |
| `splits_final_Yscaled...npz` | structurally equivalent | X max difference `0.004015088`; Y max difference `1.43e-06`           |

The source of the small numeric differences is not proven. Likely contributors include historical library/runtime differences or a different original execution environment, but this remains an inference, not a confirmed cause.

## 14. Window Reproduction

The copied window script reproduced `Tin=24`, `Tout=6`, stride `1`, target indexing immediately after the input window, masks, feature axis, site axis, and total windows `30,260`. The generated `train_windows_Tin24_Tout6_stride1.npz` is byte-for-byte identical to the frozen artifact.

**Status:** EXACTLY REPRODUCED.

## 15. Split Reproduction

The copied split script reproduced integer rounding and chronological slicing exactly:

- train: `21,182`
- validation: `4,539`
- test: `4,539`

The split artifact is byte-for-byte identical to the frozen `splits_Tin24_Tout6_stride1.npz`.

**Status:** EXACTLY REPRODUCED.

## 16. Leakage Reproduction

The reproduction preserves stride-one overlap and direct chronological slicing with no purge gap. Boundary-adjacent windows share temporal context. No leakage mitigation was introduced.

## 17. Artifact Equivalence Table

| Artifact/stage                                | Status                  | Evidence                                                     |
| --------------------------------------------- | ----------------------- | ------------------------------------------------------------ |
| Single-site current notebook feature metadata | EXACTLY REPRODUCED      | 134 names are byte-identical.                                |
| Single-site current notebook sequences        | NOT REPRODUCED          | Scratch width 134; frozen width 179; shared X values differ. |
| Single-site residual scaler parameters        | NUMERICALLY EQUIVALENT  | joblib-loaded means/scales match to machine precision.       |
| Multi-site aligned train/unseen NPZ           | EXACTLY REPRODUCED      | SHA-256 and arrays match.                                    |
| Multi-site windows                            | EXACTLY REPRODUCED      | SHA-256 and arrays match.                                    |
| Multi-site chronological split                | EXACTLY REPRODUCED      | SHA-256 and arrays match.                                    |
| 14-feature normalized split                   | STRUCTURALLY EQUIVALENT | Same keys/shapes/masks/Y; X differs slightly.                |
| 28-feature final split                        | STRUCTURALLY EQUIVALENT | Same feature order/shapes/masks; X differs slightly.         |
| Final Y-scaled split                          | STRUCTURALLY EQUIVALENT | Same schema/masks; small X/Y numeric differences.            |

## 18. Hash/Provenance Results

Scratch generated hashes and sizes are in `/tmp/stratowatch_phase3_reproduction/generated_manifest.json`. Representative hash results:

- Scratch/frozen `train_aligned.npz`: exact SHA-256 match.
- Scratch/frozen `splits_Tin24_Tout6_stride1.npz`: exact SHA-256 match.
- Scratch/frozen `feature_list.json`: exact SHA-256 match.
- Scratch/frozen single-site `sequences.npz`: different SHA-256.
- Scratch/frozen final Y-scaled NPZ: different SHA-256.
- All initially recorded production hashes remained unchanged after reproduction.

## 19. Blockers

- The frozen single-site 179-wide sequence cannot be generated by the current notebook, which produces 134 features.
- The historical 179-feature source lineage is absent from Git and not encoded in metadata.
- Final multi-site notebook outputs are not byte-identical despite identical scripted intermediates; the historical numeric environment/provenance is not recorded.
- Frozen single-site scaler bytes cannot be reproduced, although fitted parameters are equivalent through joblib loading.

## 20. UNKNOWN Items

- Exact historical feature list and ordering for the 45 unnamed single-site dimensions.
- Exact cause of frozen multi-site normalization numeric drift.
- Exact historical package versions beyond the serialized scaler's scikit-learn 1.8.0 warning.
- Whether current raw inputs were the exact inputs used for the frozen 179-wide artifact; current multi-site scripted artifacts strongly match, but no source hashes were historically recorded.

## 21. Conclusions

The scripted multi-site alignment, window, and split pipeline is reproducible exactly. The notebook-driven multi-site normalization is reproducible structurally and numerically close, but not byte-identically. The current single-site notebook reproduces the 134-name metadata and target values, but not the frozen 179-wide sequence. The residual scaler is recoverable for use through joblib and has equivalent fitted parameters in scratch.

## 22. Exact Recommended Repairs for a Future Phase

1. Preserve the frozen 179-wide sequence and reconstruct its historical feature schema before changing code.
2. Record a valid 179-name metadata artifact only after proving the feature ordering.
3. Pin and archive the environment used for scaler serialization and notebook normalization.
4. Convert notebook stages into an isolated, provenance-recording command sequence only after equivalence review.
5. Add artifact hashes and source-input hashes to a non-invasive provenance manifest.
6. Decide separately whether the split-boundary overlap is an accepted baseline property or a future research change.

## 23. Recommended Phase 4

Phase 4 should be a controlled provenance/schema resolution phase focused first on the historical single-site 179-feature lineage and second on explaining the small multi-site normalization drift. No repair should begin until those two comparisons are reviewed.
