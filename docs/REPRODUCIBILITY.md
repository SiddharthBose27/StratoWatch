# Reproducibility Guide

This document describes the commands and artifact assumptions currently present in the frozen repository. It does not add a new environment, regenerate data, or change runtime behavior.

## Repository Location

The audited workspace is:

`/Users/siddharthbose/Desktop/stratowatch`

Several existing documentation snippets use `/Users/siddharthbose/Desktop/Projects/stratowatch`; that path is stale for this workspace and should not be copied without correction.

## Frontend

The repository declares Node/Vite commands in `package.json`:

```bash
cd /Users/siddharthbose/Desktop/stratowatch
npm install
npm run dev
```

Available checks are:

```bash
npm run lint
npm run build
```

`package-lock.json` is present. Phase 1 restored dependencies with `npm install --ignore-scripts`; `npm run build` passed and `npm run lint` failed on five existing `no-explicit-any` errors in the two run pages. No UI files were changed. The install reported 13 audit vulnerabilities, and no automatic fix was applied.

## Backend

The repository documents a local FastAPI/Uvicorn process:

```bash
cd /Users/siddharthbose/Desktop/stratowatch/backend
python3 -m venv .venv
source .venv/bin/activate
python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The root README's install command uses an absolute path to the stale `Projects` location. The available dependency manifests are `requirement.txt`, `backend/requirements.txt`, `backend/stratowatch_multi_site/requirements.txt`, and the misspelled `backend/stratowatch_single_site/requiremnts.txt`. There is no single pinned Python lockfile.

## Phase 1 Verification

The new tests use only Python's standard-library test runner plus dependencies already imported by the repository's contracts:

```bash
cd /Users/siddharthbose/Desktop/stratowatch
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

These tests inspect artifacts, import model/API code, load available checkpoints, exercise tiny metric tensors, and inspect request signatures. They do not train, preprocess, regenerate, or write research artifacts.

## Required Artifacts

Single-site runtime paths expect:

- `backend/stratowatch_single_site/data/sequences.npz`
- `backend/stratowatch_single_site/data/feature_list.json` with 134 names for a sequence artifact whose feature axis is 179
- `backend/stratowatch_single_site/data/y_res_scaler.pkl`
- `backend/stratowatch_single_site/outputs/checkpoints/best_model.pt`
- `backend/stratowatch_single_site/site_1_unseen_input_data.csv` for the runner path

The current `sequences.npz`, feature list, transformer checkpoint, and residual scaler are present. The sequence feature width is 179 but the feature list contains 134 names. The residual scaler loads through `joblib.load()` and emits a scikit-learn 1.8.0-to-1.7.2 compatibility warning.

Multi-site runtime paths expect:

- `backend/stratowatch_multi_site/data/processed/splits_final_Yscaled_Tin24_Tout6_stride1.npz`
- `backend/stratowatch_multi_site/configs/target_scaler.json`
- `backend/stratowatch_multi_site/configs/site_coords_raw.npy`
- `backend/stratowatch_multi_site/data/processed/adjacency_final.npy`
- `backend/stratowatch_multi_site/outputs/checkpoints/st_transformer_best.pt`
- `backend/stratowatch_multi_site/outputs/checkpoints/graph_st_best.pt`

These are present locally. The final NPZ and ST checkpoint use 28 features, despite older documentation and the global scaler describing 14. The manifest in `configs/baseline_manifest.json` records the observed values without embedding binary contents.

## Known Pipeline Requirements

- Single-site preprocessing is notebook-based in `backend/stratowatch_single_site/preprocessing_clean.ipynb`.
- Multi-site alignment/windowing/splitting scripts are under `backend/stratowatch_multi_site/scripts/`.
- Final multi-site normalization is notebook-dependent in `backend/stratowatch_multi_site/notebooks/06_global_normalization.ipynb`.
- The current web API uses fixed preprocessed artifacts; multi-site uploaded files are accepted but not used to construct inference data.
- Model training commands exist in the two project READMEs, but training was intentionally not run in Phase 1.

## Known Missing or Unverified Pieces

- The single-site residual scaler requires its intended Joblib loader and a compatible scikit-learn version.
- No separate dynamic-wind graph checkpoint exists.
- There is no CI or deployment configuration.
- No dependency installation or version lock was added by Phase 1.
- The frontend checks require the declared Node dependencies to be installed before they can run.
