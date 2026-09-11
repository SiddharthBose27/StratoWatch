# Data Pipeline Map

This is a static Phase 2 lineage audit. No notebook cell was executed and no production artifact was regenerated.

## Single-site lineage

```text
site_1_train_data.csv
  -> preprocessing_clean.ipynb
  -> datetime index and hourly reindex
  -> missingness flags, interpolation, ffill/bfill
  -> calendar, wind, lag, rolling, and satellite features
  -> grouped feature scaling
  -> residual targets: target - forecast
  -> residual StandardScaler
  -> 24-step sliding sequences
  -> data/sequences.npz + data/feature_list.json + data/y_res_scaler.pkl
  -> train.py / evaluate.py / baseline scripts
```

The source notebook is `backend/stratowatch_single_site/preprocessing_clean.ipynb`. Its raw inputs are `backend/stratowatch_single_site/site_1_train_data.csv` and `site_1_unseen_input_data.csv`. The observed train input has 16 columns: four time fields, five forecast/meteorological fields, three satellite fields, and two targets. The unseen input has the same columns without targets.

The notebook constructs `datetime` from `year`, `month`, `day`, and `hour`, sorts by it, and reindexes each dataframe to an hourly range. It adds missingness indicators before cleaning, replaces infinities with missing values, interpolates in time, then applies forward/backward fill. It computes cyclic time features, wind speed/direction, forecast lags at 1/2/3/6/12/24 hours, and rolling mean/std features at windows 3/6/12/24.

Residual targets are `O3_target - O3_forecast` and `NO2_target - NO2_forecast`. The notebook fits a residual `sklearn.preprocessing.StandardScaler`, saves it through `joblib.dump`, and creates sequences with `Tin=24`; each target row is the row immediately after the input window. `train.py` and `evaluate.py` consume `data/sequences.npz`; evaluation inverse-scales residuals and reconstructs `forecast + residual`.

The current artifacts do not fully match the notebook metadata: `X_seq` has width 179, while `feature_list.json` has 134 names. The notebook code would save the current dataframe columns, so the exact run that produced both current files is not proven.

## Multi-site lineage

```text
data/raw/data/*.csv
  -> scripts/01_unpack_and_inspect.py
  -> scripts/02_check_all_sites.py
  -> scripts/03_align_multisite.py
  -> train_aligned.npz + unseen_aligned.npz + aligned_meta.json
  -> scripts/04_make_windows.py
  -> train_windows_Tin24_Tout6_stride1.npz
  -> scripts/05_split_windows.py
  -> splits_Tin24_Tout6_stride1.npz
  -> notebooks/06_global_normalization.ipynb
  -> normalized/enhanced/geo/final feature artifacts
  -> final_feature_scaler.json + splits_final_Tin24_Tout6_stride1.npz
  -> target_scaler.json + splits_final_Yscaled_Tin24_Tout6_stride1.npz
  -> src/data/dataset.py
  -> STTransformer / GraphSTTransformer
```

`03_align_multisite.py` preserves all train columns except `O3_target` and `NO2_target` in X, forms independent train/unseen union timelines, reindexes each site, sets missing numeric values to zero, and writes masks that preserve original validity. `04_make_windows.py` uses 24 input steps, six target steps, and stride one. `05_split_windows.py` slices windows chronologically at 70/15/15.

The final feature additions are notebook-only: six cyclic calendar fields, six satellite observation/age fields, and two broadcast coordinate fields. The final partial scaler and target scaler are also notebook-generated. Runtime loaders use the final Y-scaled NPZ; the active model checkpoints record 28 input features.

## Active runtime artifacts

- Final input/target artifact: `backend/stratowatch_multi_site/data/processed/splits_final_Yscaled_Tin24_Tout6_stride1.npz`.
- Input feature metadata: `backend/stratowatch_multi_site/configs/final_feature_scaler.json`.
- Target metadata: `backend/stratowatch_multi_site/configs/target_scaler.json`.
- Loader: `backend/stratowatch_multi_site/src/data/dataset.py`.
- Model inputs: `backend/stratowatch_multi_site/src/models/st_transformer.py` and `graph_st_transformer.py`.

## Lineage limitations

There is no hash-linked manifest connecting the current raw CSVs to the current final NPZ, no one-command pipeline, and no recorded notebook execution environment. The final target-scaling notebook cell contains a stale absolute project path, so clean-environment reproduction is not currently demonstrated.
