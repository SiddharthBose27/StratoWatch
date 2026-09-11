# StratoWatch Phase 0 Baseline Audit

**Audit scope:** inspection and documentation only. No application source, model, preprocessing, UI, API, dependency, or data files were modified.

**Repository inspected:** `/Users/siddharthbose/Desktop/stratowatch`

**Baseline date:** 2026-09-06

## A. Repository Structure

### Top-level

- `src/`: React/Vite frontend source.
- `public/`: static frontend assets, including example plots.
- `backend/app/`: FastAPI application and synchronous model runners.
- `backend/stratowatch_single_site/`: single-station preprocessing artifacts, transformer, baselines, training, evaluation, and plots.
- `backend/stratowatch_multi_site/`: multi-station preprocessing scripts, notebooks, datasets, graph/transformer models, training, evaluation, and reports.
- `package.json`, `package-lock.json`: frontend dependencies and scripts.
- `requirement.txt`: root Python dependency list used by the documented backend setup.
- `backend/requirements.txt`: backend/API dependency list.
- `backend/stratowatch_multi_site/requirements.txt`: multi-site research dependency list.
- `backend/stratowatch_single_site/requiremnts.txt`: single-site dependency list; the filename itself is misspelled.
- `README.md`: project overview, setup, architecture, and claimed workflows.
- `vite.config.ts`, `tsconfig*.json`, `tailwind.config.js`, `postcss.config.js`, `eslint.config.js`: frontend/tooling configuration.

### Repository hygiene

`.gitignore` excludes CSV/XLSX data, NPZ/NPY artifacts, PyTorch checkpoints, pickled scalers, environments, and build output. The current workspace contains ignored runtime artifacts, including processed multi-site arrays, adjacency matrices, checkpoints, single-site sequences, scalers, and uploads. A clean checkout therefore requires a separate artifact-provisioning step before model execution can be assumed to work.

No test suite, CI workflow, Dockerfile, Docker Compose file, Procfile, or hosting configuration was found.

## B. Frontend Architecture

### Entry point and routing

- `src/main.tsx`: mounts React in `StrictMode`, wraps the application in `BrowserRouter`, and imports global CSS.
- `src/App.tsx`: defines routes for `/`, `/single-site`, `/multi-site`, `/methodology`, and `/docs`.
- `src/components/SiteShell.tsx`: global page shell and footer.
- `src/components/Navbar.tsx`: fixed navigation, route links, and contact CTA.
- `src/styles/ui.ts`: shared UI styling helpers.
- `src/index.css`: global/Tailwind styling and animations.
- `src/App.css`: additional application styles, including some legacy Vite/demo styling.

### Pages

- `src/pages/Home.tsx`: product overview, claimed capabilities, and hard-coded performance snapshot.
- `src/pages/SingleSite.tsx`: single-file upload, model selection, request submission, metrics, and base64 plot rendering.
- `src/pages/MultiSite.tsx`: multi-file selection, site-count selection, model selection, warnings, metrics, and plot rendering.
- `src/pages/Methodology.tsx`: static description of data, architecture, metrics, results, and limitations.
- `src/pages/Docs.tsx`: static developer documentation, setup snippets, configuration references, and endpoint table.

### Frontend/API contract

`src/pages/SingleSite.tsx` posts `file` and `model_name` to `/api/single-site/run`.

`src/pages/MultiSite.tsx` posts repeated `files`, `site_count`, and `model_name` to `/api/multi-site/run`. The UI requires the selected file count to equal the requested site count, but the current backend does not consume the uploaded file contents.

Both pages optionally use `VITE_API_BASE`; otherwise they use relative API paths. They expect JSON containing `ok`, `metrics`, optional `warning`, optional `logs`, and base64-encoded PNG plot objects.

### Frontend configuration

- `vite.config.ts`: development proxy for `/api` and `/health` to `http://127.0.0.1:8000`.
- `package.json`: Vite dev/build/preview scripts and ESLint script; no frontend test script.
- `eslint.config.js`: ESLint, TypeScript, React Hooks, and React Refresh configuration.
- `tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json`: TypeScript project configuration.
- `tailwind.config.js`, `postcss.config.js`: styling toolchain configuration.

## C. Backend Architecture

### API application

- `backend/app/main.py`: creates the FastAPI application, configures local CORS, and exposes three routes:
  - `GET /health`: returns `{"ok": true}`.
  - `POST /api/single-site/run`: accepts one uploaded file and `model_name`, stores the upload, and calls `run_single_site_real`.
  - `POST /api/multi-site/run`: accepts `site_count`, `model_name`, and a list of files, then calls `run_multisite`.

CORS permits only `http://localhost:5173` and `http://127.0.0.1:5173`. There is no authentication, rate limiting, request-size limit, database, job queue, or persistent job state.

### Single-site runner

- `backend/app/runner.py`: maps frontend model names to Python scripts/modules, copies the uploaded file to the fixed `backend/stratowatch_single_site/site_1_unseen_input_data.csv`, runs a synchronous subprocess, parses printed metrics with regular expressions, optionally runs confusion-matrix generation, and returns PNGs as base64.

Important current behavior:

- The upload path in `backend/app/main.py` is built using `file.filename` directly.
- Shared model outputs and plots are read after execution.
- Subprocesses have no timeout.
- Concurrent requests can overwrite shared inputs and outputs.
- Model stdout/stderr tails are returned to the frontend.

### Multi-site runner

- `backend/app/multi_runner.py`: uses a fixed processed NPZ at `backend/stratowatch_multi_site/data/processed/splits_final_Yscaled_Tin24_Tout6_stride1.npz`, a fixed target scaler, fixed figure directories, and fixed evaluator modules.
- For `site_count == 7`, it runs checkpoint-based evaluation.
- For any other site count, it computes a persistence baseline from the fixed NPZ and returns a fallback warning.
- It clears shared figure directories before each request.
- It invokes `backend/stratowatch_multi_site/src/report_plots.py` in a subprocess with a 30-second plotting timeout.

The multi-site uploaded files are accepted by the API signature but are explicitly unused. Runtime inference therefore reflects the preprocessed local artifact, not the files selected in the UI.

## D. ML Architecture

### Single-site model family

- `backend/stratowatch_single_site/models/temporal_transformer.py`: temporal transformer used by `train.py` and `evaluate.py`.
- `backend/stratowatch_single_site/train.py`: trains a residual transformer on `data/sequences.npz`, uses a chronological 70/15/15 split, MSE loss, AdamW, ReduceLROnPlateau, early stopping, and saves a checkpoint and loss curve.
- `backend/stratowatch_single_site/evaluate.py`: loads the transformer checkpoint and residual scaler, predicts residuals, reconstructs targets as `forecast + residual`, computes model and forecast-only metrics, and saves prediction/summary CSVs.

Baselines:

- `backend/stratowatch_single_site/baselines/train_xgb.py`: XGBoost residual regressors for O3 and NO2.
- `backend/stratowatch_single_site/baselines/train_rf.py`: Random Forest residual baseline.
- `backend/stratowatch_single_site/baselines/train_lstm.py`: LSTM residual model.
- `backend/stratowatch_single_site/baselines/train_tcn.py`: temporal convolutional residual model.
- `backend/stratowatch_single_site/baselines/common_data.py`: shared sequence loader, chronological split, flattening, and forecast extraction.
- `backend/stratowatch_single_site/baselines/confusion_matrices.py`: categorical severity-bin confusion matrices.

Observed baseline contract issue: `train_rf.py` requests `X_train_flat` and `X_test_flat`, while `common_data.py` returns `X_train` and `X_test`. The Random Forest path should be treated as unverified until this is audited.

### Multi-site model family

- `backend/stratowatch_multi_site/src/models/st_transformer.py`: baseline spatio-temporal transformer. It projects features, applies per-site temporal attention, cross-site spatial attention, and a direct multi-horizon head.
- `backend/stratowatch_multi_site/src/models/graph_st_transformer.py`: graph-enhanced model with temporal encoding, adjacency propagation, spatial attention, and a multi-horizon head.
- `backend/stratowatch_multi_site/src/data/dataset.py`: loads train/validation/test tensors and masks from the final NPZ schema.
- `backend/stratowatch_multi_site/src/utils/metrics.py`: masked MAE, MSE, RMSE, and Huber functions.

The documented/current multi-site shape assumptions are seven sites, 14 input features, two targets, 24 input hours, and six forecast hours. The model derives some dimensions from the NPZ but checkpoints and graph artifacts remain schema-dependent.

## E. Data Pipeline

### Single-site preprocessing

- `backend/stratowatch_single_site/preprocessing_clean.ipynb`: notebook-based cleaning, hourly time reindexing, missing-value treatment, feature engineering, scaling, and sequence generation.
- `backend/stratowatch_single_site/data/feature_list.json`: feature ordering used to find `O3_forecast` and `NO2_forecast`.
- `backend/stratowatch_single_site/data/sequences.npz`: runtime sequence artifact; ignored by Git.
- `backend/stratowatch_single_site/data/y_res_scaler.pkl`: residual scaler; ignored by Git.

The runtime scripts assume the preprocessed `sequences.npz` already exists. The API upload does not regenerate this artifact, so arbitrary uploaded schema/content is not fully validated or transformed through the documented preprocessing flow.

### Multi-site preprocessing

- `backend/stratowatch_multi_site/scripts/01_unpack_and_inspect.py`: input archive inspection.
- `backend/stratowatch_multi_site/scripts/02_check_all_sites.py`: checks expected train/unseen files.
- `backend/stratowatch_multi_site/scripts/03_align_multisite.py`: reads site CSVs, constructs datetime indexes, uses the union timeline, preserves non-target columns as features, zero-imputes missing numeric values, and writes masks.
- `backend/stratowatch_multi_site/scripts/04_make_windows.py`: creates stride-one windows with `TIN=24` and `TOUT=6`.
- `backend/stratowatch_multi_site/scripts/05_split_windows.py`: chronological 70/15/15 window split.
- `backend/stratowatch_multi_site/notebooks/06_global_normalization.ipynb`: additional normalization/final dataset stage.
- `backend/stratowatch_multi_site/configs/global_scaler.json`, `final_feature_scaler.json`, and `target_scaler.json`: scaling metadata.

The command-line scripts produce intermediate names such as `splits_Tin24_Tout6_stride1.npz`, while runtime and evaluation use `splits_final_Yscaled_Tin24_Tout6_stride1.npz`. The final transformation is therefore partly notebook-driven and is not represented as one reproducible command-line pipeline.

## F. Multi-site Pipeline

The intended sequence is:

1. Raw site CSVs in `backend/stratowatch_multi_site/data/raw/data/`.
2. Alignment and masks from `scripts/03_align_multisite.py`.
3. Sliding windows from `scripts/04_make_windows.py`.
4. Chronological split from `scripts/05_split_windows.py`.
5. Final target normalization from `notebooks/06_global_normalization.ipynb`.
6. PyTorch loading through `src/data/dataset.py`.
7. Training through `src/train_baseline.py` or `src/train_graph.py`.
8. Evaluation through `src/eval_baseline_realunits.py` or `src/eval_graph_realunits.py`.
9. API execution through `backend/app/multi_runner.py`.

`src/train_baseline.py` saves `outputs/checkpoints/st_transformer_best.pt`. `src/train_graph.py` saves `outputs/checkpoints/graph_st_best.pt`. Training prefers Apple MPS when available and otherwise uses CPU.

The web path is currently a fixed-artifact evaluation path, not a complete uploaded multi-site inference pipeline.

## G. Graph Pipeline

- `backend/stratowatch_multi_site/scripts/06_save_raw_site_coords.py`: writes manually specified raw latitude/longitude coordinates for seven sites.
- `backend/stratowatch_multi_site/data/processed/adjacency_final.npy`: active training/evaluation adjacency artifact; ignored by Git.
- `backend/stratowatch_multi_site/src/models/graph_st_transformer.py`: computes wind-conditioned adjacency and graph propagation.
- `backend/stratowatch_multi_site/src/train_graph.py`: trains the graph model with the fixed adjacency and coordinates.
- `backend/stratowatch_multi_site/src/eval_graph_realunits.py`: loads `graph_st_best.pt`, adjacency, coordinates, and target scaler for real-unit test evaluation.

The graph model hard-codes wind feature indices `8` and `9`, assumes coordinate ordering matches the seven-site dataset, uses a fixed `beta=0.5` wind adjustment, and applies dynamic wind-conditioned adjacency inside the same model implementation.

The API maps both `graph_st_static` and `graph_st_dynamic_wind` to `src.eval_graph_realunits`. `src/report_plots.py` looks for a separate dynamic checkpoint but falls back to `graph_st_best.pt` when one is absent. Thus the UI presents distinct static and dynamic choices, but the existence of distinct trained/evaluated artifacts is not confirmed.

## H. Evaluation Pipeline

### Single-site evaluation

- `backend/stratowatch_single_site/evaluate.py`: transformer residual evaluation and forecast-only comparison.
- `backend/stratowatch_single_site/baselines/train_xgb.py`, `train_lstm.py`, and `train_tcn.py`: training plus real-unit reconstruction and metrics.
- `backend/stratowatch_single_site/baselines/confusion_matrices.py`: severity-bin evaluation.
- `backend/stratowatch_single_site/compare_experiments.py`: compares stored experiment summaries.
- `backend/stratowatch_single_site/make_table.py`: reads stored metrics for tables.

### Multi-site evaluation

- `backend/stratowatch_multi_site/src/eval_baseline.py`: scaled baseline checkpoint evaluation.
- `backend/stratowatch_multi_site/src/eval_baseline_realunits.py`: inverse-scaled baseline evaluation.
- `backend/stratowatch_multi_site/src/eval_graph_realunits.py`: inverse-scaled graph evaluation.
- `backend/app/multi_runner.py`: selects evaluation mode and parses printed MAE/RMSE.

Multi-site real-unit evaluators average batch-level metric values rather than computing one globally weighted aggregate, which should be preserved as a baseline behavior until metric semantics are explicitly audited.

## I. Visualization Pipeline

### Single-site visualization

- `backend/stratowatch_single_site/plots.py`: paper-style prediction/residual plots from stored CSVs.
- `backend/stratowatch_single_site/baselines/confusion_matrices.py`: confusion-matrix PNGs.
- `backend/stratowatch_single_site/outputs/plots/`: generated single-site plots.
- `backend/stratowatch_single_site/outputs/confusion_matrices/`: generated classification plots.
- `public/assets/plots/`: static plots displayed by the frontend methodology/content pages.

### Multi-site visualization

- `backend/stratowatch_multi_site/src/plot_predictions.py`: prediction visualization.
- `backend/stratowatch_multi_site/src/report_plots.py`: UI report pack including horizon metrics, per-site MAE, residual distribution, O3 severity confusion, PCA proxy, and sample prediction plots.
- `backend/stratowatch_multi_site/notebooks/final_analysis.ipynb`: notebook-based analysis and plots.
- `backend/stratowatch_multi_site/outputs/figures/`: generated figures.
- `backend/stratowatch_multi_site/outputs/figures/ui_report/`: API-facing report figures.

`report_plots.py` intentionally evaluates only a limited number of batches (`max_batches=6`) for responsiveness. UI report metrics may therefore differ from full test-set evaluator metrics.

## J. Current Testing Status

No automated test files or test directories were found. There is no configured pytest, Vitest, Jest, API contract, preprocessing, dataset-schema, model-shape, or end-to-end test suite. No CI configuration was found.

Available repository checks are:

- `npm run lint` in `package.json`.
- `npm run build` in `package.json`.
- Manual model, notebook, and API execution scripts.

During this audit, `npm run lint` could not start because the local `eslint` executable was unavailable. The chained build command therefore did not run. No dependencies were installed, per the Phase 0 constraints. Python model execution was not attempted because it would mutate shared outputs and potentially run long-running model code.

## K. Current Deployment Configuration

The repository describes a local development deployment:

- Frontend: Vite development server from `package.json`.
- Backend: documented Uvicorn command targeting `app.main:app` in `backend/app/main.py`.
- Frontend proxy: `vite.config.ts` forwards `/api` and `/health` to `127.0.0.1:8000`.
- CORS: local Vite origins only, configured in `backend/app/main.py`.
- Optional frontend API base: `VITE_API_BASE` in `src/pages/SingleSite.tsx` and `src/pages/MultiSite.tsx`.
- Runtime storage: local filesystem under `backend/uploads/` and model output directories.
- Persistence: none; no database or object storage.
- Background execution: none; model subprocesses run synchronously in request handling.
- Production packaging: no container, process manager, CI/CD, cloud hosting, or production ASGI configuration found.

Documentation paths in `README.md` and `src/pages/Docs.tsx` refer to `/Users/siddharthbose/Desktop/Projects/stratowatch`, which differs from the audited workspace path.

## L. Potential Hard-coded Assumptions

- Seven multi-site stations and manually ordered site coordinates.
- Delhi coordinate values in `scripts/06_save_raw_site_coords.py`.
- 24-hour input window and six-hour forecast horizon.
- Two targets: `O3_target` and `NO2_target`.
- Wind feature indices `8` and `9`.
- Fixed NPZ, scaler, adjacency, checkpoint, and output filenames.
- Fixed single-site destination filename `site_1_unseen_input_data.csv`.
- Chronological 70/15/15 splitting and stride-one overlapping windows.
- Relative working-directory assumptions in research scripts.
- Localhost ports and CORS origins.
- O3 severity thresholds `[0, 60, 120, 180]` in `src/report_plots.py`.
- Apple MPS preference with CPU fallback.
- Shared mutable output directories across requests.
- Optional `VITE_API_BASE` as the only documented environment override.
- Uploaded file extensions are selected in the browser, but content/schema validation is not implemented in the API.

## M. Technical Debt

### Reproducibility

- Final multi-site normalization is notebook-dependent.
- Required data, checkpoints, scalers, and arrays are ignored by Git.
- Dependency manifests overlap but are inconsistent; `requiremnts.txt` is misspelled.
- Documentation references stale absolute paths and artifact names.

### Correctness and contract drift

- Multi-site uploads are not connected to preprocessing or inference.
- Random Forest baseline dictionary keys do not match `common_data.py`.
- Static and dynamic graph selections share an evaluator and may share a checkpoint.
- UI claims and stored static plots are not necessarily generated by the same runtime path.
- UI report metrics use a partial batch sample.
- The split script's “without leakage” claim should be audited because adjacent stride-one windows can share temporal context across split boundaries.

### Operational and security risk

- Direct use of uploaded filenames creates path traversal and collision risk.
- Synchronous subprocess execution has no general timeout or cancellation.
- Shared files and directories are vulnerable to concurrent-request races.
- No authentication, rate limiting, request-size limit, or isolation exists.
- Backend logs can expose paths and implementation details in API responses.

## N. Files High-risk to Modify

These files control runtime contracts, artifact schemas, or model behavior and require coordinated tests and artifact review:

- `backend/app/main.py`
- `backend/app/runner.py`
- `backend/app/multi_runner.py`
- `backend/stratowatch_multi_site/src/models/st_transformer.py`
- `backend/stratowatch_multi_site/src/models/graph_st_transformer.py`
- `backend/stratowatch_multi_site/src/data/dataset.py`
- `backend/stratowatch_multi_site/src/train_baseline.py`
- `backend/stratowatch_multi_site/src/train_graph.py`
- `backend/stratowatch_multi_site/src/eval_baseline_realunits.py`
- `backend/stratowatch_multi_site/src/eval_graph_realunits.py`
- `backend/stratowatch_multi_site/scripts/03_align_multisite.py`
- `backend/stratowatch_multi_site/scripts/04_make_windows.py`
- `backend/stratowatch_multi_site/scripts/05_split_windows.py`
- `backend/stratowatch_multi_site/notebooks/06_global_normalization.ipynb`
- `backend/stratowatch_single_site/preprocessing_clean.ipynb`
- `backend/stratowatch_single_site/data/feature_list.json`
- `backend/stratowatch_single_site/baselines/common_data.py`
- `backend/stratowatch_single_site/evaluate.py`
- `vite.config.ts` and the request logic in `src/pages/SingleSite.tsx` and `src/pages/MultiSite.tsx`

## O. Files That Appear Safer to Modify

These are relatively isolated documentation or presentation surfaces, provided their claims are kept aligned with the frozen runtime:

- `README.md`
- `backend/stratowatch_single_site/README.md`
- `backend/stratowatch_multi_site/README.md`
- `src/pages/Docs.tsx`
- `src/pages/Methodology.tsx`
- `src/pages/Home.tsx`
- `src/components/Navbar.tsx`
- `src/components/SiteShell.tsx`
- `src/styles/ui.ts`
- `src/App.css` and `src/index.css` for isolated presentation-only changes

Even these files should not be changed in the baseline freeze unless the change is explicitly documentation-only or presentation-only.

## P. Recommended Order of Future Changes

1. Freeze and archive the current runtime artifacts, dependency versions, representative inputs, current outputs, and exact local commands.
2. Add read-only schema and smoke tests for API response shape, dataset keys/shapes, checkpoint loading, and metric calculations.
3. Resolve documentation and artifact-path drift without changing runtime behavior.
4. Make preprocessing reproducible from scripts, including the final normalization stage, while preserving current outputs for comparison.
5. Define and test the upload contract before connecting uploaded files to model inference.
6. Isolate request-scoped files and outputs, add safe filename handling, and introduce bounded/background execution.
7. Audit and repair model/data contracts, including the Random Forest loader mismatch and graph static/dynamic distinction.
8. Re-run the frozen baseline on the same artifacts and compare metrics/plots before any architecture or preprocessing change.
9. Only after the baseline is reproducible, consider model, graph, feature, UI, or deployment changes in separate phases.

# CURRENT BASELINE

## What currently works

- The repository contains a Vite/React frontend with routes for the main product, single-site, multi-site, methodology, and documentation views.
- The FastAPI application defines health, single-site, and multi-site endpoints.
- The local frontend proxy and documented Uvicorn integration are configured for development.
- The single-site runner can invoke its configured scripts when the required ignored artifacts and dependencies are present.
- The multi-site runner can evaluate the fixed seven-site checkpoint path or compute a fixed-NPZ persistence fallback for other site counts.
- Training, evaluation, and visualization scripts exist for both single-site and multi-site research workflows.
- Existing generated plots and metrics provide a visible research/demo baseline.

## What is uncertain

- Whether the current ignored checkpoints, scalers, arrays, and local Python environment are mutually compatible.
- Whether the deployed instance matches the checked-out repository and local artifacts.
- Whether all documented model variants have distinct checkpoints and behavior.
- Whether arbitrary uploaded CSV/XLSX files satisfy the required schemas.
- Whether the current API survives concurrent requests or large/slow model runs.
- Whether reported UI metrics match full-test metrics.
- Whether the Random Forest baseline currently executes successfully.
- Whether the final notebook normalization steps can be reproduced exactly outside the current workspace.

## What must not be changed until audited

- NPZ keys, tensor shapes, feature ordering, target ordering, and scaler semantics.
- Site ordering, coordinate mapping, adjacency artifacts, and wind feature indices.
- Window sizes, stride, chronological split behavior, and mask semantics.
- Checkpoint filenames and model constructor dimensions.
- Single-site residual reconstruction: `forecast + residual`.
- API field names, route names, runner model mappings, and response keys.
- Shared output paths relied on by the current UI.
- Static/dynamic graph model selection behavior.
- Existing generated artifacts used as the current visual or numerical reference.

## Recommended next phase

**Phase 1 should be a reproducibility and contract-verification phase.** Preserve this report and the current artifacts, then add non-invasive tests and manifests that prove the current data schemas, model loading, API responses, and evaluation metrics before changing application behavior or model architecture.
