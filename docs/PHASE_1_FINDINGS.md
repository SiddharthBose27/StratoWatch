# Phase 1 Findings

**Scope:** reproducibility and contract verification for the frozen StratoWatch baseline. No model, preprocessing, UI, API, dependency, checkpoint, scaler, dataset, or generated output was changed.

## 1. Verified

- `GET /health`, `POST /api/single-site/run`, and `POST /api/multi-site/run` are defined in `backend/app/main.py`.
- Single-site request parameters are `file` and `model_name`.
- Multi-site request parameters are `files`, `site_count`, and `model_name`.
- `backend/stratowatch_single_site/data/sequences.npz` exists with `X_seq=(43489, 24, 179)` and `y_seq=(43489, 2)`.
- `backend/stratowatch_single_site/data/feature_list.json` exists with 134 entries and contains `O3_forecast` and `NO2_forecast`.
- `backend/stratowatch_multi_site/data/processed/splits_final_Yscaled_Tin24_Tout6_stride1.npz` exists with train/validation/test partitions, masks, seven sites, two targets, `tin=24`, `tout=6`, and 28 input features.
- `backend/stratowatch_multi_site/data/processed/adjacency_final.npy` exists with shape `(7, 7)`.
- `backend/stratowatch_multi_site/configs/site_coords_raw.npy` exists with shape `(7, 2)`.
- `st_transformer_best.pt` and `graph_st_best.pt` exist and load as checkpoint containers. The ST checkpoint records `num_features=28`, `num_sites=7`, `tin=24`, and `tout=6`.
- Single-site transformer, ST Transformer, and Graph ST Transformer constructors import and their available checkpoints load in the verification tests.
- Metric functions implement masked global sums divided by masked counts, with RMSE as the square root of masked MSE and Huber `delta=1.0`.
- Multi-site evaluators add one scalar metric per batch and divide by the number of batches, so batch sizes do not receive global mask-weighted aggregation.
- Single-site reconstruction remains `forecast + predicted_residual` in `backend/stratowatch_single_site/evaluate.py`.

## 2. Failed Verification

- `backend/stratowatch_single_site/data/y_res_scaler.pkl` is a valid Joblib-serialized `StandardScaler`; direct `pickle.load()` is inappropriate and reports an invalid load key. `joblib.load()` succeeds with a scikit-learn 1.8.0-to-1.7.2 `InconsistentVersionWarning`.
- `backend/stratowatch_single_site/data/sequences.npz` has feature width 179, while `feature_list.json` has 134 names. The mismatch is now captured by a failing contract test.
- The Random Forest path is runtime-breaking at the data contract boundary: `train_rf.py` requests `X_train_flat` and `X_test_flat`, while `common_data.py` returns the already-flattened keys `X_train` and `X_test`. The expected result is a `KeyError` before model construction.
- The Phase 0 description of 14 multi-site input features does not match the active final artifact, which has 28 features. `global_scaler.json` describes 14 features, while `final_feature_scaler.json`, the final NPZ, and the ST checkpoint describe 28.
- `graph_st_static` and `graph_st_dynamic_wind` are not independently verified as distinct behaviors. The API maps both to `src.eval_graph_realunits`, the graph implementation is wind-conditioned, and no separate dynamic checkpoint is present. `report_plots.py` falls back to `graph_st_best.pt` for dynamic mode.
- After installing the declared lockfile dependencies with `npm install --ignore-scripts`, `npm run build` passed. `npm run lint` failed with five existing `@typescript-eslint/no-explicit-any` errors in `src/pages/SingleSite.tsx` and `src/pages/MultiSite.tsx`.

## 3. Missing Artifacts

- No separate dynamic-wind graph checkpoint was found under `backend/stratowatch_multi_site/outputs/checkpoints/`.
- The single-site residual scaler is loadable with its intended Joblib loader, but its serialization environment differs from the current environment.
- No committed test runner or pytest dependency exists. Phase 1 tests use Python's standard-library `unittest` instead.
- No CI, container, production process, or deployment manifest exists.

## 4. Contract Mismatches

| Contract                     | Observed mismatch                                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Multi-site feature count     | Phase 0/docs describe 14; active final NPZ and ST checkpoint use 28.                                                     |
| Single-site residual scaler  | Valid Joblib `StandardScaler`; direct raw-pickle loading is not the correct loader.                                      |
| Single-site feature metadata | `X_seq` has width 179; `feature_list.json` has 134 entries.                                                              |
| Random Forest loader         | Consumer requests `X_train_flat`/`X_test_flat`; provider returns `X_train`/`X_test`.                                     |
| Static/dynamic graph modes   | UI/API expose two names, but both use the same evaluator and graph class; dynamic checkpoint is absent.                  |
| Multi-site upload behavior   | Files are accepted by `main.py` but ignored by `multi_runner.py`; fixed NPZ is used.                                     |
| Documentation paths          | `README.md` and `src/pages/Docs.tsx` reference `/Users/siddharthbose/Desktop/Projects/stratowatch`, not this repository. |
| Artifact names               | Docs mention `feature_scaler.json` and a config adjacency path that differ from active names/paths.                      |

## 5. Existing Technical Debt Confirmed

- Required runtime artifacts are ignored by Git and are not provisioned by a repository command.
- Final multi-site normalization is partly notebook-dependent.
- Dependency manifests overlap without a single pinned Python environment; `requiremnts.txt` is misspelled.
- `backend/requirements.txt` does not explicitly list `joblib` or `seaborn`, although single-site scripts use them.
- Root `requirement.txt` includes API and ML dependencies but uses broad/unpinned versions.
- `npm install --ignore-scripts` completed and reported 13 package audit vulnerabilities: 2 low, 1 moderate, and 10 high. No audit fix was applied.
- API runners use shared mutable files/directories and synchronous subprocesses.
- Uploaded single-site filenames are used directly in save paths.
- UI report figures are generated from only a limited number of test batches.

## 6. Risks

- A clean checkout cannot reproduce model execution without separately supplied ignored artifacts.
- The residual scaler requires Joblib loading and emits a scikit-learn version warning under the current environment.
- The 14-versus-28 feature discrepancy can cause checkpoint/data/schema incompatibility if either side is changed without an artifact audit.
- Static/dynamic graph result labels may not correspond to distinct runtime behavior.
- Multi-site users can select files in the UI while the backend evaluates unrelated fixed local data.
- Concurrent API requests can overwrite each other's inputs and outputs.

## 7. Changes Deliberately Deferred to Later Phases

- Repairing or regenerating `y_res_scaler.pkl`.
- Renaming Random Forest keys or changing loader behavior.
- Changing the multi-site feature schema or normalization pipeline.
- Introducing a separate static graph implementation or dynamic checkpoint.
- Connecting uploaded multi-site files to preprocessing/inference.
- Fixing upload path safety, request isolation, timeouts, authentication, or queues.
- Updating model architecture, preprocessing, metrics, split behavior, UI, or API semantics.
- Upgrading or pinning dependencies.

All of these are findings or follow-up work, not Phase 1 fixes.
