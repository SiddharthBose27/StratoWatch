# Phase 7 Data Pipeline Specification

**Status:** COMPLETE

**Phase:** Phase 7 — Research-Grade Data Contract + Reproducible Preprocessing

**Repository:** `/Users/siddharthbose/Desktop/stratowatch`

**Branch:** `stratowatch-2.0`

**Implementation Date:** 2026-09-06

---

## 1. Overview

Phase 7 implements the research-grade data contract and reproducible preprocessing foundation specified in Phase 6.

The completed implementation provides:

* Explicit dataset, schema, and preprocessing versioning
* Authoritative feature schemas with deterministic ordering
* Raw-data validation
* Leakage-controlled temporal splitting with purge separation
* Train-only feature and target scaling
* Deterministic temporal window generation
* Explicit input and target masks
* Metadata-bearing research artifacts
* Configuration serialization and reload support
* Deterministic CLI execution
* Separation of the reproducible research pipeline from historical legacy artifacts
* Run-to-run reproducibility verification

The official reproducible single-site research baseline uses:

**134 input features → 24-hour historical context → 6-hour multi-horizon forecast**

The historical 179-feature single-site artifacts remain preserved as legacy reference material and are not silently replaced or reconstructed.

---

# 2. Architecture

## 2.1 Package Structure

```text
backend/stratowatch_data/

├── __init__.py
├── __main__.py

├── config/
│   ├── __init__.py
│   └── base_config.py

├── validation/
│   ├── __init__.py
│   └── raw_data_validator.py

├── schema/
│   ├── __init__.py
│   └── feature_schema.py

├── preprocessing/
│   ├── __init__.py
│   ├── single_site_pipeline.py
│   ├── multi_site_pipeline.py
│   └── feature_engineering.py

├── splitting/
│   ├── __init__.py
│   └── temporal_split.py

├── scaling/
│   ├── __init__.py
│   └── scaler_fitter.py

├── windowing/
│   ├── __init__.py
│   └── window_generator.py

├── artifacts/
│   ├── __init__.py
│   ├── artifact_writer.py
│   └── manifest_generator.py

├── utils/
│   ├── __init__.py
│   └── time_utils.py

└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_schema.py
    ├── test_splitting.py
    ├── test_windowing.py
    ├── test_reproducibility.py
    └── run_tests.py
```

## 2.2 Module Responsibilities

| Module          | Responsibility                                            |
| --------------- | --------------------------------------------------------- |
| `config`        | Pipeline configuration, versioning, serialization         |
| `validation`    | Raw-data validation and quality checks                    |
| `schema`        | Feature/target definitions and ordering                   |
| `preprocessing` | End-to-end single-site and multi-site pipelines           |
| `splitting`     | Chronological splitting and purge separation              |
| `scaling`       | Train-only scaler fitting and transformation              |
| `windowing`     | Deterministic temporal window generation                  |
| `artifacts`     | Research artifact and metadata generation                 |
| `utils`         | Time and deterministic utility functions                  |
| `tests`         | Contract, splitting, windowing, and reproducibility tests |

---

# 3. Input Contract

## 3.1 Single-Site Raw Data

The single-site pipeline consumes the established project training and unseen-input files.

Required timestamp fields:

* `year`
* `month`
* `day`
* `hour`

Forecast variables:

* `O3_forecast`
* `NO2_forecast`
* `T_forecast`
* `q_forecast`
* `u_forecast`
* `v_forecast`
* `w_forecast`

Satellite variables:

* `NO2_satellite`
* `HCHO_satellite`
* `ratio_satellite`

Training data additionally contains:

* `O3_target`
* `NO2_target`

The training data is used to construct residual targets:

```text
O3_residual  = O3_target  - O3_forecast
NO2_residual = NO2_target - NO2_forecast
```

## 3.2 Validation Rules

The raw-data validator checks:

* Required columns
* Target-column presence where applicable
* Timestamp construction
* Duplicate timestamps
* Chronological ordering
* Infinite values
* Missingness
* File existence
* Basic temporal metadata

Invalid raw-data conditions are surfaced before downstream preprocessing.

---

# 4. Official Single-Site Feature Contract

## 4.1 Feature Count

The official reproducible single-site feature schema contains:

**134 features**

The authoritative ordering is taken from:

```text
backend/stratowatch_single_site/data/feature_list.json
```

The Phase 7 pipeline preserves this ordering exactly.

## 4.2 Feature Groups

The 134 features consist of:

### Base and forecast features

* Timestamp components
* O3 forecast
* NO2 forecast
* Meteorological forecast variables

### Missingness indicators

Missingness flags are retained for the relevant raw variables, including satellite observations.

### Calendar features

* `hour_sin`
* `hour_cos`
* `month_sin`
* `month_cos`
* `dow`
* `dow_sin`
* `dow_cos`

### Wind features

* `wind_speed_h`
* `wind_dir_rad`
* `wind_dir_sin`
* `wind_dir_cos`

### Lag features

For the seven forecast variables:

```text
1, 2, 3, 6, 12, 24 hours
```

### Rolling statistics

For the seven forecast variables:

```text
windows = 3, 6, 12, 24
```

with:

* rolling mean
* rolling standard deviation

The exact feature identity and ordering are controlled by the authoritative feature-list artifact rather than inferred alphabetically.

---

# 5. Historical 179-Feature Baseline

The historical single-site branch contains frozen 179-feature sequence artifacts.

These artifacts are preserved as historical reference material.

The exact 179-feature schema could not be fully recovered from the available provenance evidence. Therefore:

* The 179-feature artifact is preserved
* No unsupported feature identities are invented
* No silent mapping to the 134-feature schema is performed
* The 134-feature branch is treated as the official reproducible research baseline

This distinction is fundamental to the research record.

```text
Historical baseline:
179 features
        ↓
Preserved reference artifact

Reproducible research baseline:
134 features
        ↓
Phase 7 data contract
```

The two branches must not be treated as numerically equivalent datasets.

---

# 6. Window Contract

Phase 7 standardizes the research pipeline around multi-horizon forecasting.

## 6.1 Parameters

```text
Tin    = 24 hours
Tout   = 6 hours
Stride = 1 hour
```

Therefore:

```text
Input:
X[t : t+24]

Target:
Y[t+24 : t+30]
```

The resulting tensor contract is:

```text
X: (N, 24, 134)

Y: (N, 6, 2)
```

where the two target channels represent:

```text
O3_residual
NO2_residual
```

## 6.2 Deterministic Window Generation

The window generator:

* Uses chronological ordering
* Uses fixed `Tin`, `Tout`, and `stride`
* Excludes incomplete windows
* Propagates X/Y masks
* Records window indices
* Produces deterministic output

The completed Phase 7 single-site pipeline generated:

```text
X_windows: (25052, 24, 134)
Y_windows: (25052, 6, 2)
```

---

# 7. Leakage-Controlled Temporal Splitting

## 7.1 Purge Requirement

For:

```text
Tin  = 24
Tout = 6
```

a window beginning at `t` occupies:

```text
Input:  [t, t+23]
Target: [t+24, t+29]
```

Therefore the complete window occupies:

```text
[t, t+29]
```

To ensure two windows do not share timestamps, a later window must begin at:

```text
t_next >= t + 30
```

Therefore:

```text
minimum separation = Tin + Tout = 30
```

The configured purge separation is:

```text
30
```

## 7.2 Split Contract

The final single-site run produced:

```text
Train X: (17536, 24, 134)
Val X:   (3757, 24, 134)
Test X:  (3699, 24, 134)

Train Y: (17536, 6, 2)
Val Y:   (3757, 6, 2)
Test Y:  (3699, 6, 2)
```

The split is chronological and purge-aware.

The pipeline explicitly verifies the research split contract before writing artifacts.

---

# 8. Mask Contract

Phase 7 preserves missingness information through explicit masks.

## 8.1 Input Masks

```text
X_mask_train
X_mask_val
X_mask_test
```

with shapes:

```text
(N, 24, 134)
```

## 8.2 Target Masks

```text
Y_mask_train
Y_mask_val
Y_mask_test
```

with shapes:

```text
(N, 6, 2)
```

Masks are aligned with their corresponding arrays.

All generated masks contain valid binary mask values.

---

# 9. Scaling Policy

## 9.1 Train-Only Fitting

Scaling is fitted exclusively on training data.

Validation and test data are transformed using the already-fitted scalers.

No validation or test observations are used to fit scaling statistics.

## 9.2 Single-Site Feature Scalers

The completed single-site pipeline produces separate scaler artifacts for:

```text
time_raw
time_cyclic
forecast_met
pollut_fore
```

The target scaler is stored separately.

Each scaler has:

* PKL serialized representation
* JSON metadata
* Schema association
* Fitting-scope metadata
* Scaler parameters
* Creation metadata

## 9.3 Target Scaling

Residual targets are scaled using a dedicated target scaler.

The target channels are:

```text
O3_residual
NO2_residual
```

The fitting scope is:

```text
train_only
```

---

# 10. Research Artifact Contract

The completed pipeline writes a versioned research artifact package.

Representative single-site artifacts include:

```text
stratowatch_single_v2_scaled.npz

stratowatch_single_v2_feature_names.json
stratowatch_single_v2_config.json
stratowatch_single_v2_scaled_metadata.json
schema_v1_schema.json

stratowatch_single_v2_time_raw_scaler.pkl
stratowatch_single_v2_time_raw_scaler.json

stratowatch_single_v2_time_cyclic_scaler.pkl
stratowatch_single_v2_time_cyclic_scaler.json

stratowatch_single_v2_forecast_met_scaler.pkl
stratowatch_single_v2_forecast_met_scaler.json

stratowatch_single_v2_pollut_fore_scaler.pkl
stratowatch_single_v2_pollut_fore_scaler.json

stratowatch_single_v2_target_scaler.pkl
stratowatch_single_v2_target_scaler.json
```

## 10.1 Final Artifact Shapes

The validated artifact contains:

```text
X_train       (17536, 24, 134)
X_val         (3757, 24, 134)
X_test        (3699, 24, 134)

Y_train       (17536, 6, 2)
Y_val         (3757, 6, 2)
Y_test        (3699, 6, 2)

X_mask_train  (17536, 24, 134)
X_mask_val    (3757, 24, 134)
X_mask_test   (3699, 24, 134)

Y_mask_train  (17536, 6, 2)
Y_mask_val    (3757, 6, 2)
Y_mask_test   (3699, 6, 2)
```

All generated arrays passed finite-value validation.

---

# 11. Provenance and Reproducibility

Phase 7 requires deterministic execution from a fixed configuration and input dataset.

## 11.1 Deterministic Components

The following components were verified as deterministic:

* Configuration serialization
* Feature ordering
* Feature engineering
* Window generation
* Temporal splitting
* Scaling
* Mask generation
* Artifact contents

## 11.2 Run A vs Run B Verification

Two independent executions were performed:

```text
/tmp/stratowatch_phase7_runA
/tmp/stratowatch_phase7_runB
```

### Array comparison

All X/Y arrays were:

```text
shape = identical
values = identical
```

This included:

* X train
* X validation
* X test
* Y train
* Y validation
* Y test
* X masks
* Y masks

### Feature schema

```text
Feature count A: 134
Feature count B: 134
Feature names identical: True
```

### Configuration

```text
Config identical: True
```

### Scaler metadata

All five scaler metadata artifacts were identical after excluding expected creation timestamps:

```text
time_raw:       True
time_cyclic:    True
forecast_met:   True
pollut_fore:    True
target:         True
```

Final result:

```text
PASS: All scaler metadata are reproducible.
PASS: PHASE 7 REPRODUCIBILITY CHECK PASSED
```

Therefore the Phase 7 pipeline has demonstrated exact run-to-run reproducibility for the validated artifact package.

---

# 12. CLI Interface

The pipeline is executable through:

```bash
python3 -m stratowatch_data
```

Single-site execution:

```bash
python3 -m stratowatch_data \
  --pipeline-type single_site \
  --output-dir /tmp/stratowatch_phase7_final
```

Important configuration parameters include:

```text
--pipeline-type
--config
--dataset-version
--output-dir
--raw-data-dir
--tin
--tout
--stride
--train-ratio
--val-ratio
--test-ratio
--purge-gap
--site-id
--scaler-type
--random-seed
--scale-targets
--no-scale-targets
```

The final validated configuration uses:

```text
pipeline_type = single_site
dataset_version = stratowatch_single_v2
Tin = 24
Tout = 6
stride = 1
purge_gap = 30
scaler_type = standard
```

---

# 13. Versioning Strategy

## 13.1 Dataset Version

Current single-site research dataset:

```text
stratowatch_single_v2
```

## 13.2 Schema Version

The research schema is explicitly versioned and serialized with the generated artifacts.

Feature ordering is immutable within a schema version.

## 13.3 Version Mutation Rules

A new schema version is required when any of the following changes:

* Feature order
* Feature identity
* Feature transformation
* Source columns
* Scaling policy
* Feature dimensionality
* Target definition

A new dataset version is required when:

* Raw data changes
* Schema changes
* Preprocessing changes
* Split policy changes

This prevents silent dataset mutation.

---

# 14. Legacy Integration

Phase 7 deliberately preserves historical project artifacts.

## 14.1 Preserved Legacy Artifacts

The following remain separate:

* Historical 179-feature single-site sequences
* Legacy 134-feature single-site artifacts
* Legacy multi-site artifacts
* Historical scalers
* Historical checkpoints

No legacy artifact is silently overwritten by the research pipeline.

## 14.2 Research Artifacts

Research artifacts are written separately and versioned through their dataset identifiers.

This creates a clean separation between:

```text
Historical / legacy evidence
```

and:

```text
Reproducible research baseline
```

---

# 15. Final Validation Status

The following Phase 7 research contracts were successfully verified:

* [x] Raw data validation
* [x] Authoritative 134-feature schema
* [x] Exact feature ordering
* [x] 24-hour input window
* [x] 6-hour prediction horizon
* [x] Deterministic window generation
* [x] Purge-aware temporal split
* [x] Input masks
* [x] Target masks
* [x] Train-only scaling
* [x] Scaler metadata
* [x] Finite-value checks
* [x] Artifact generation
* [x] Configuration serialization
* [x] Run A / Run B array reproducibility
* [x] Run A / Run B schema reproducibility
* [x] Run A / Run B configuration reproducibility
* [x] Run A / Run B scaler reproducibility
* [x] Legacy artifact preservation

---

# 16. Known Limitations

## 16.1 Historical 179-Feature Schema

The exact identity and ordering of all 179 historical features could not be recovered from the available project evidence.

This is documented rather than reconstructed speculatively.

The reproducible research pipeline therefore uses the independently verified 134-feature schema.

## 16.2 Timezone

The raw project data follows the existing timezone-naive convention.

Phase 7 does not introduce timezone conversion.

This remains a documented limitation for future productionization.

## 16.3 Model Layer

Phase 7 is a data-contract and preprocessing phase.

It does not establish final model performance.

Model architecture, training, ablation studies, evaluation, and production inference remain Phase 8 responsibilities.

---

# 17. Phase 8 Readiness

Phase 7 establishes the data foundation required for Phase 8.

## Ready

* [x] Versioned research dataset
* [x] Explicit feature schema
* [x] Deterministic feature ordering
* [x] 24-hour input context
* [x] 6-hour forecast horizon
* [x] Purge-aware temporal split
* [x] Train-only scaling
* [x] Missingness masks
* [x] Reproducible artifacts
* [x] Configuration contract
* [x] Provenance metadata
* [x] Run-to-run reproducibility verification
* [x] Legacy baseline preservation

## Phase 8 Responsibilities

Phase 8 will implement and evaluate:

* Final temporal model
* Spatial/graph component
* Static graph mode
* Dynamic wind-conditioned graph mode
* Residual forecasting
* Multi-horizon prediction
* Mask-aware training and evaluation
* Baseline comparisons
* Ablation experiments
* Error analysis
* Model checkpoint provenance
* API integration
* Frontend/product integration

---

# 18. Conclusion

**Phase 7 is complete.**

The project now has a reproducible research-grade preprocessing foundation built around an explicitly versioned **134-feature single-site schema**, a **24-hour input context**, and a **6-hour multi-horizon residual forecasting target**.

The pipeline has demonstrated:

1. **Explicit Data Contracts** — Feature and target definitions are versioned and serialized.

2. **Leakage Control** — Temporal splitting uses a 30-hour minimum window separation, and scalers are fitted only on training data.

3. **Deterministic Processing** — Feature engineering, windowing, splitting, scaling, and artifact generation are reproducible.

4. **Exact Reproducibility** — Independent Run A and Run B executions produced identical arrays, feature schemas, configuration, and scaler metadata.

5. **Scientific Integrity** — The unresolved historical 179-feature schema is explicitly preserved rather than reconstructed through unsupported assumptions.

6. **Legacy Preservation** — Historical artifacts remain available for reference and comparison.

7. **Phase 8 Readiness** — The data layer is now sufficiently defined and reproducible to support final model development and rigorous evaluation.

**Phase 7 — RESEARCH-GRADE DATA CONTRACT + REPRODUCIBLE PREPROCESSING: COMPLETE ✅**

The next development stage is **Phase 8 — Final Model + Evaluation + Product Integration**.
