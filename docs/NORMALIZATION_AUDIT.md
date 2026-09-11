# Normalization Audit

## Stage 1: 14-feature global normalization

`backend/stratowatch_multi_site/notebooks/06_global_normalization.ipynb` first loads `splits_Tin24_Tout6_stride1.npz`, which contains the 14 raw non-target features. It fits `fit_global_standard_scaler(X_train, X_mask_train)` using valid masked training-window values only. For each feature, mean and standard deviation are computed over flattened `(N, Tin, S)` positions; empty features use neutral `(0, 1)` values and near-zero standard deviations are replaced with `1`.

The fitted parameters are applied to `X_train`, `X_val`, and `X_test`. Masks are copied unchanged and Y is not scaled in this stage. The result is `splits_normalized_Tin24_Tout6_stride1.npz` and `configs/global_scaler.json`.

## Stage 2: feature engineering

The notebook reloads the unnormalized split file to compute calendar features from original `year`, `month`, `day`, and `hour` values. It appends six cyclical features to the globally normalized 14-feature X arrays. It then derives satellite observation flags and within-window age values from raw arrays and raw masks, appending six more features with all-one masks. Finally, it broadcasts raw site latitude/longitude values across each sample, time step, and site, appending two more all-valid features.

The result is 28 features. This stage is notebook-only; no equivalent script exists in `backend/stratowatch_multi_site/scripts/`.

## Stage 3: final partial feature scaling

Notebook cell 15 fits a mask-aware train-only scaler on exactly:

- `NO2_satellite_age`
- `HCHO_satellite_age`
- `ratio_satellite_age`
- `site_lat`
- `site_lon`

Non-selected features use mean `0` and standard deviation `1` in the metadata and therefore pass through. The scaler is applied to train, validation, and test X arrays. X masks are unchanged. The output is `splits_final_Tin24_Tout6_stride1.npz` and `configs/final_feature_scaler.json`.

## Stage 4: target scaling

Notebook cell 16 loads the final unscaled-Y NPZ and fits a target mean/std from `Y_train` using `Y_mask_train` only. It applies those parameters to Y train/validation/test, leaves Y masks unchanged, and writes `splits_final_Yscaled_Tin24_Tout6_stride1.npz` plus `configs/target_scaler.json`.

Observed target metadata:

| Target     |              Mean |               Std |
| ---------- | ----------------: | ----------------: |
| O3_target  | 31.96749496459961 | 34.72024154663086 |
| NO2_target | 39.13277053833008 | 29.90866470336914 |

The runtime evaluators inverse-transform with `y * std + mean`.

## Mask interaction

- Missing raw X/Y values are zero-filled before scaling, while masks retain validity.
- Global and partial feature-scaler fitting excludes masked values.
- Target-scaler fitting excludes masked target values.
- Engineered masks are set to one because those values are computed, not raw observations.
- Current model losses/metrics use Y masks. Current ST and graph forward paths do not use X masks for attention or graph computation.

## Provenance limitations

The final target-scaling cell contains a stale absolute `PROJECT_ROOT` pointing to an external `Luftwächter(multisite_transformer)` directory. The current active final artifacts exist under this repository, but the exact execution that generated them is not reproducible from that cell without path adaptation. No path adaptation was performed in Phase 2.

The 14-feature `global_scaler.json` is not a scaler for the active 28-feature final model input. It belongs to the earlier global-normalization stage. The active runtime uses the final Y-scaled NPZ and target scaler; the feature scaler is preserved as provenance metadata rather than loaded by the runtime evaluators.

## Status

Feature and target scaling behavior is **PARTIALLY VERIFIED** from static notebook code and persisted metadata. Exact artifact equivalence from current raw files remains **UNKNOWN** because the notebook was not executed and no provenance hashes exist.
