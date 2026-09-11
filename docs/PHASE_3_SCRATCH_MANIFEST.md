# Phase 3 Scratch Manifest

## Scratch directory

`/tmp/stratowatch_phase3_reproduction/`

The scratch directory is outside the repository and was not copied into the workspace. It remains available for inspection and can be removed independently.

## Copied inputs and source

- `source_copy/preprocessing_clean.ipynb`
- `source_copy/06_global_normalization.ipynb`
- `source_copy/multi_scripts/01_unpack_and_inspect.py` through `06_save_raw_site_coords.py`
- `multi_site/raw/` including copied raw CSVs, `data.zip`, and coordinate text
- `single_project/site_1_train_data.csv`
- `single_project/site_1_unseen_input_data.csv`

The initial manifest also hashed the frozen production artifacts, all multi-site raw CSV/TXT inputs, relevant scripts/notebooks, scalers, adjacency, coordinates, and checkpoints. It is stored at `/tmp/stratowatch_phase3_reproduction/baseline_manifest.json`.

## Commands executed

1. Read-only SHA-256/size/timestamp manifest generation.
2. Copied multi-site scripts/raw inputs into `multisite_project`.
3. Ran copied `01_unpack_and_inspect.py`, `02_check_all_sites.py`, `03_align_multisite.py`, `04_make_windows.py`, and `05_split_windows.py`.
4. Executed normalization notebook cells 1, 2, 3, 5, 6, 7, 15, and 16 in a scratch namespace.
5. Executed single-site notebook cells 0 through 12 in `single_project`.
6. Compared scratch/frozen NPZ arrays, JSON metadata, hashes, shapes, and deterministic statistics.
7. Loaded residual scalers with `joblib.load` and recorded version warnings.
8. Rehashed all initially recorded production files after reproduction.

## Path substitutions

- Multi-site notebook `PROJECT_ROOT` assignments were replaced only in the in-memory scratch execution namespace with the scratch project path.
- Single-site notebook relative paths were left unchanged; copied inputs were placed at the expected scratch working directory.
- No production source path was redirected to a production output location.

## Generated temporary artifacts

The scratch pipeline generated aligned NPZs, windows, split NPZs, normalized NPZs, final feature/target scaler JSONs, a valid scratch residual scaler, logs, comparison reports, and `generated_manifest.json`. The generated manifest records 14 scratch files with sizes and SHA-256 hashes.

## Cleanup status

- Repository generated artifacts: none.
- Production research artifacts: unchanged; all initial hashes matched after reproduction.
- Scratch artifacts: retained outside the repository for audit evidence.
- No packages were installed or repository dependency manifests changed during Phase 3.
