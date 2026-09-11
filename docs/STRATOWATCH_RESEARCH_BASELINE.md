# StratoWatch Research Baseline

**Status:** Phase 6 canonical research contract

**Repository:** `/Users/siddharthbose/Desktop/stratowatch`

**Branch at Phase 6 start:** `stratowatch-2.0`

**Commit at Phase 6 start:** `5e5ef4bed0a2118a78e0ceb7e4696aa0f38adf6e`

This document freezes the evidence-backed baseline before StratoWatch 2.0 research development. It supplements, and does not replace, the detailed Phase 0-5 reports.

## 1. Executive Summary

### Historical baseline

The historical single-site branch is a **179-feature historical model lineage**.

Status: **HISTORICAL / PARTIALLY DOCUMENTED / NOT FULLY REPRODUCIBLE**.

Evidence:

- `backend/stratowatch_single_site/data/sequences.npz` has `X_seq=(43489,24,179)` and `y_seq=(43489,2)`.
- `experiments/full_model/sequences.npz` and `experiments/tuned_model/sequences.npz` are byte-identical to the active 179-wide sequence.
- Corresponding active/full/tuned checkpoints expect 179 input features.
- The exact identities and ordering of 45 additional historical dimensions cannot be recovered.

The historical artifacts remain frozen for comparison and provenance. They are not a reproducible source for new feature engineering.

### Reproducible baseline

The official reproducible single-site baseline is the **134-feature current pipeline**:

- Source: `backend/stratowatch_single_site/preprocessing_clean.ipynb`.
- Output: `X_seq=(43489,24,134)`, `y_seq=(43489,2)`.
- Metadata: `backend/stratowatch_single_site/data/feature_list.json`.
- Closest archived experiment: `backend/stratowatch_single_site/experiments/no_satellite/sequences.npz`.

The official reproducible multi-site baseline is the **28-feature final pipeline**:

- Seven sites.
- `Tin=24` hours.
- `Tout=6` hours.
- Two targets: O3 and NO2.
- Final Y-scaled artifact: `backend/stratowatch_multi_site/data/processed/splits_final_Yscaled_Tin24_Tout6_stride1.npz`.
- Historical environment reproduction demonstrated byte-equivalent final artifacts; environment caveats remain because the exact original execution environment is not independently archived.

## 2. Baseline Classification

| Baseline                 | Domain      | Feature width | Input history |          Forecast horizon | Artifact                                                                                              | Reproducibility                              | Scientific status                          | Allowed future use                     |
| ------------------------ | ----------- | ------------: | ------------: | ------------------------: | ----------------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------ | -------------------------------------- |
| Historical single-site   | Single-site |           179 |      24 hours | Next-step residual target | `backend/stratowatch_single_site/data/sequences.npz`                                                  | Historical reference; not fully reproducible | Frozen historical evidence                 | Historical comparison only             |
| Reproducible single-site | Single-site |           134 |      24 hours | Next-step residual target | `backend/stratowatch_single_site/experiments/no_satellite/sequences.npz` and current notebook lineage | Reproducible current baseline                | Official reproducible single-site baseline | New controlled single-site experiments |
| Multi-site final         | Multi-site  |            28 |      24 hours |                   6 hours | `backend/stratowatch_multi_site/data/processed/splits_final_Yscaled_Tin24_Tout6_stride1.npz`          | Reproducible with environment caveats        | Official reproducible multi-site baseline  | New controlled multi-site experiments  |

The historical 179-feature artifact must never be silently substituted for the 134-feature reproducible baseline, and the 134-feature pipeline must never be presented as a reconstruction of the 179-feature schema.

## 3. Historical 179-Feature Baseline Policy

The 179-feature branch is frozen.

It may be used for:

- preserving historical results;
- comparing previously obtained metrics;
- documenting experiment provenance;
- compatibility checks against the existing 179-input checkpoints.

It may not be used as the source of truth for new feature engineering because its complete schema is unknown.

The following are forbidden:

- renaming unknown features;
- inferring feature names from correlations;
- claiming the 179 dimensions are the current 134 features plus 45 known features;
- reconstructing a synthetic 179-feature dataset;
- overwriting or resaving the historical artifact;
- retraining the historical branch with a guessed schema.

Historical numerical artifacts are preserved as historical evidence, not treated as a reproducible preprocessing pipeline.

## 4. Official Research Baseline Decision

### Official reproducibility baseline

Future research development must use the documented 134-feature single-site pipeline and verified 28-feature multi-site pipeline as the reproducible foundations.

### Historical reference baseline

The 179-feature single-site artifacts remain a historical reference baseline only. Their original results must retain their original provenance and must not be recomputed under a guessed schema.

## 5. Data Contract

### Temporal resolution

Hourly data.

### Window geometry

- Input history: `Tin=24` hours.
- Forecast horizon: `Tout=6` hours for the multi-site baseline.
- Current multi-site legacy stride: `1`.
- Single-site current notebook sequence length: `24`.

### Targets

The pollutant targets are:

- `O3` / `O3_target`.
- `NO2` / `NO2_target`.

Target ordering must be explicitly stored and must not be inferred from array position alone.

### Spatial setting

Single-site and multi-site experiments are separate experimental domains. A single-site sequence must not be treated as a multi-site tensor, and a multi-site artifact must not be reduced to a single-site experiment without an explicit experiment definition.

### Site identity

Site ordering must be obtained from explicit configuration or artifact metadata. Future implementations must not rely silently on array position, filesystem ordering, or hardcoded site numbers. Any coordinate, adjacency, mask, and tensor site axis must use the same recorded mapping.

## 6. Feature Schema Versioning

Every future feature schema is a versioned model contract and must record:

- schema version;
- ordered feature list;
- feature count;
- source column or source artifact for each feature;
- transformation description;
- units where applicable;
- missing-value behavior;
- scaling behavior;
- target ordering;
- site ordering where spatial features are present.

A checkpoint is not reproducible without its corresponding feature schema. Feature names, feature ordering, preprocessing configuration, target ordering, and site mapping together define the model input contract.

The unresolved historical 179 schema is not a valid schema source for future work.

## 7. Missing Data Contract

### Current baseline behavior

The current baselines preserve their existing behavior:

- single-site notebook: missingness flags, interpolation, forward fill, and backward fill;
- multi-site alignment: masks are created before zero imputation, then missing values are filled with zero;
- multi-site engineered features receive documented masks;
- target masks are used by multi-site losses/metrics.

This behavior is frozen for historical interpretation.

### Research-grade future policy

Future experiments must:

- represent missingness explicitly;
- document every imputation operation;
- preserve masks where applicable;
- prevent future observations from influencing a prediction-time imputation;
- ensure interpolation and filling are causally valid for the forecast timestamp;
- record whether an operation is fit globally, per site, or per split.

A change from the current behavior requires a new preprocessing/schema version and a new experiment identifier.

## 8. Scaling and Normalization Contract

Every scaler record must specify:

- scaler type;
- feature order;
- target order;
- fitted-data scope;
- mask policy;
- serialization format;
- package/environment metadata;
- artifact identifier or hash.

Future normalization statistics must be fitted using training data only. Validation and test observations must never influence learned normalization statistics.

The baseline multi-site pipeline contains an intermediate 14-feature global scaler and a final 28-feature partial feature scaler, followed by train-only target scaling. Inverse target scaling uses the recorded target mean/std and reconstructs original-unit predictions before final reporting.

The single-site residual scaler is a valid Joblib-serialized scikit-learn `StandardScaler`; its loader and serialization environment must be recorded in future experiment manifests.

## 9. Data Split and Leakage Policy

### Legacy baseline split

The historical multi-site baseline uses stride-one windows and chronological 70/15/15 slicing. Adjacent windows overlap, including around split boundaries. This is labeled:

**LEGACY BASELINE SPLIT**

Its behavior is preserved and must not be silently changed when interpreting historical results.

### Research-grade split

Future experiments must use temporally separated partitions. A training sample must not share input or forecast timestamps with validation or test samples.

For a window starting at time index $t$:

- input timestamps are $[t, t+Tin-1]$;
- forecast timestamps are $[t+Tin, t+Tin+Tout-1]$.

For `Tin=24` and `Tout=6`, a later partition's first start index $s$ must satisfy:

$$s \ge t + Tin + Tout = t + 30$$

relative to the last training start $t$ to make the full input-plus-forecast timestamp ranges disjoint. With stride one, this means excluding the intervening starts $t+1$ through $t+29$, subject to the exact split implementation. This is the minimum geometry-derived separation; a larger embargo may be required by the scientific design.

This future policy is labeled:

**RESEARCH-GRADE SPLIT**

No current dataset was changed in Phase 6.

## 10. Evaluation Contract

Future official evaluation must report:

- O3 metrics separately;
- NO2 metrics separately;
- aggregate metrics only where scientifically meaningful;
- original physical units;
- observation masks;
- horizon-wise performance;
- target-wise performance;
- site-wise performance for multi-site experiments;
- split boundaries and purge/embargo policy.

For valid observations with error $e_i=\hat{y}_i-y_i$ and mask $m_i$:

$$N_{valid}=\sum_i m_i$$

$$MAE=\frac{\sum_i m_i|e_i|}{N_{valid}}$$

$$RMSE=\sqrt{\frac{\sum_i m_i e_i^2}{N_{valid}}}$$

R² must be computed only where its denominator and target variance are mathematically appropriate, with the aggregation scope stated.

### Existing evaluator behavior

The current multi-site evaluators compute masked metrics per batch and average the resulting scalar values across batches. This is a legacy behavior and must not be silently reinterpreted as a globally valid weighted aggregate.

### Future research evaluator

The future evaluator must accumulate valid observations globally, then compute MAE and RMSE from the global masked sums. It must not average batch RMSE values. Any change from the legacy evaluator requires a new evaluator version and documented comparison.

## 11. Baseline Model Family

Future experiments may compare:

### Classical / tabular baselines

- Random Forest.
- XGBoost.

### Sequential deep-learning baselines

- LSTM.
- TCN.

### Transformer baseline

- Spatio-Temporal Transformer.

### Graph-enhanced models

- Static Graph + Transformer.
- Dynamic/Wind-conditioned Graph + Transformer.

### Proposed research direction

A graph-aware spatio-temporal residual forecasting architecture.

No architecture is declared superior until controlled experiments demonstrate that result.

## 12. Configuration-First Policy

Future experiments must be driven by versioned configuration rather than source edits. The configuration contract must include at minimum:

- dataset identifier and hash;
- site mapping;
- feature schema version;
- targets and target ordering;
- `Tin`, `Tout`, and stride;
- split policy and purge gap;
- scaler identifiers;
- model and graph mode;
- hidden dimension, layers, attention heads, and dropout;
- learning rate, batch size, epochs, early stopping;
- random seed;
- checkpoint policy.

## 13. Experiment Reproducibility Contract

Every future experiment must record:

1. experiment ID;
2. timestamp;
3. Git commit;
4. dataset hash;
5. raw-data hashes;
6. feature-schema hash;
7. preprocessing version;
8. complete configuration;
9. environment/package versions;
10. random seed;
11. model architecture;
12. checkpoint hash;
13. training metrics;
14. validation metrics;
15. test metrics;
16. evaluation code version;
17. notes and warnings.

The manifest must identify exactly which data, code, configuration, and environment produced the result.

## 14. Randomness Policy

Future experiments must explicitly control and record Python, NumPy, PyTorch, and CUDA randomness where applicable. If deterministic execution is not guaranteed, the experiment manifest must state that limitation. Perfect reproducibility must not be claimed without verification.

## 15. Checkpoint Contract

Future checkpoints must contain or reference:

- architecture version;
- feature schema version;
- target schema;
- input dimensions;
- site count and site mapping;
- `Tin` and `Tout`;
- graph mode;
- full configuration;
- training commit;
- dataset identifier/hash;
- scaler identifier/hash;
- environment metadata.

Existing checkpoints remain unchanged and are not retroactively declared fully self-describing.

## 16. Evaluation and Reporting Requirements

Every future report must include:

### Dataset

- date range;
- sites and ordering;
- sample counts;
- missingness;
- split boundaries;
- purge/embargo gap.

### Model

- architecture;
- parameter count;
- graph type;
- temporal mechanism;
- residual mechanism.

### Training

- optimizer;
- learning rate;
- batch size;
- epochs;
- early stopping;
- seed.

### Results

- O3 MAE/RMSE/R²;
- NO2 MAE/RMSE/R²;
- horizon-wise metrics;
- aggregate metrics;
- baseline comparison.

### Reproducibility

- Git commit;
- configuration;
- environment;
- dataset hash;
- checkpoint hash.

## 17. Research Questions

- **RQ1:** Does residual learning improve short-term O3 and NO2 forecasting compared with direct prediction?
- **RQ2:** Does spatial information improve forecasting compared with single-site temporal models?
- **RQ3:** Does graph-aware spatial modeling improve over a non-graph spatio-temporal Transformer?
- **RQ4:** Does dynamically wind-conditioned connectivity improve over a static spatial graph?
- **RQ5:** How does performance change across forecast horizons?

These are testable questions, not claims about expected winners.

## 18. Ablation Plan

### Core experiments

1. Persistence/simple baseline.
2. XGBoost.
3. Random Forest.
4. LSTM.
5. TCN.
6. Transformer.
7. Static Graph Transformer.
8. Dynamic/Wind Graph Transformer.
9. Residual + Transformer.
10. Residual + Static Graph Transformer.
11. Residual + Dynamic Graph Transformer.

### Optional ablations

- Feature-family removal.
- Satellite observation/age features.
- Coordinate features.
- Static versus dynamic adjacency.
- Mask-aware versus documented alternative handling.
- Forecast horizon and context-window sensitivity.

Computational constraints may limit optional ablations. No ablations are run in Phase 6.

## 19. Scientific Integrity Policy

StratoWatch research work must enforce:

- no fabricated metrics;
- no invented feature names;
- no invented historical preprocessing;
- no retrospective baseline modification to improve results;
- no cherry-picking test results;
- no unrecorded evaluation-method changes after observing results;
- preservation of historical result provenance;
- reproducibility requirements for new experiments;
- honest reporting of uncertainty and limitations.

## 20. Research and Production Separation

### Layer 1 - Research data/model pipeline

Preprocessing, training, evaluation, experiments, and checkpoints.

### Layer 2 - Inference service

Validated inputs, model loading, inference, and standardized outputs.

### Layer 3 - Frontend

User interaction, visualization, and presentation of model outputs.

Future work must avoid embedding research preprocessing or evaluation logic directly inside frontend/API code.

## 21. Verified Technical Debt

### Research correctness

- The historical 179-feature schema and ordering remain unresolved.
- The current single-site notebook produces 134 features while historical full/tuned artifacts use 179.
- The legacy multi-site split has overlapping context across partition boundaries.
- Existing multi-site evaluator aggregation averages batch-level metric scalars.
- Static and dynamic graph paths are not fully distinguished in the current implementation.
- The report plot path evaluates only a limited batch count (`max_batches=6`).

### Engineering / production debt

- Random Forest runtime key mismatch: consumer expects `X_train_flat`/`X_test_flat`, loader returns `X_train`/`X_test`.
- Frontend lint has existing `no-explicit-any` errors.
- npm audit reported 13 vulnerabilities in the Phase 1 environment.
- Graph code contains hardcoded wind indices and site assumptions.
- API/model execution is synchronous and uses shared mutable runtime artifacts.
- No complete automated test suite or CI workflow.
- No container or production deployment configuration.
- Stale absolute paths and artifact-name drift exist in documentation.

No issue in this list is fixed by Phase 6.

## 22. What Must Not Change Silently

The following require a new version, documentation, experiment identifier, and reproducibility record:

- feature ordering or feature count;
- target ordering;
- site ordering or coordinate mapping;
- train/validation/test boundaries;
- scaler fitting scope;
- missing-data handling;
- window geometry;
- forecast horizon;
- graph construction;
- metric aggregation;
- baseline definitions;
- checkpoint compatibility.

## 23. Future Phase Roadmap

- **Phase 7:** Research-grade data contract and reproducible preprocessing implementation.
- **Phase 8:** Leakage-controlled dataset generation and versioned feature schema.
- **Phase 9:** Baseline model cleanup and reproducible training pipeline.
- **Phase 10:** StratoWatch 2.0 spatio-temporal architecture implementation.
- **Phase 11:** Static/dynamic graph implementation and controlled ablations.
- **Phase 12:** Evaluation framework and statistical comparison.
- **Phase 13:** Experiment tracking and reproducibility packaging.
- **Phase 14:** Inference/API hardening.
- **Phase 15:** Frontend/product integration.
- **Phase 16:** Paper results, figures, tables, and final scientific validation.

Phase 7 requires explicit authorization and is not started by this document.
