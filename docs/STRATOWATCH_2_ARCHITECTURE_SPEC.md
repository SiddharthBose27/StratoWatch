# StratoWatch 2.0 Architecture Specification

**Status:** specification only; no implementation is authorized by this document.

## 1. Problem Definition

StratoWatch forecasts short-term ground-level O3 and NO2 using historical meteorological, forecast, satellite/reanalysis, and ground-station information. The research problem has two explicitly separated settings:

- single-site residual forecasting;
- multi-site spatio-temporal forecasting across an explicitly configured station set.

The reproducible foundations are the current 134-feature single-site lineage and the verified 28-feature multi-site lineage. The historical 179-feature single-site branch is a frozen reference only.

## 2. Data Flow

```text
raw sources
  -> versioned data contract
  -> causal preprocessing and masks
  -> versioned feature schema
  -> train-only normalization
  -> leakage-controlled temporal split
  -> model input tensors
  -> training/checkpoint manifest
  -> original-unit evaluation
  -> experiment report
```

Research preprocessing must remain in the research pipeline. The inference service consumes validated, versioned artifacts rather than embedding notebook logic.

## 3. Feature and Schema Contract

Every model input must reference:

- feature schema version;
- ordered feature names;
- feature count;
- source and transformation for each feature;
- units where applicable;
- missingness/mask behavior;
- scaler identifier and fitted-data scope;
- target order;
- site mapping.

A feature tensor without this metadata is not a reproducible model input. The unresolved 179-feature historical schema cannot be used for new feature engineering.

## 4. Temporal Encoder

The temporal component is Transformer-based and operates over the hourly history for each site. The specification requires:

- explicit `Tin` from configuration;
- causal/forecast-valid input construction;
- documented positional encoding;
- attention and layer counts from configuration;
- mask semantics defined by the data contract;
- no unrecorded feature-order assumptions.

The initial reproducible baseline uses `Tin=24`; future experiments may vary it only under a new configuration and experiment ID.

## 5. Spatial Graph Module

The spatial component operates across the explicitly configured site set. It may combine message passing and/or spatial attention. The graph contract must specify:

- site ordering;
- coordinate source;
- adjacency source;
- adjacency normalization;
- self-loop policy;
- directionality;
- edge weighting;
- graph tensor shape;
- whether masks affect propagation.

No site count, coordinate order, adjacency order, or edge semantics may be inferred silently from array positions.

## 6. Dynamic Graph Module

The dynamic graph variant may condition connectivity on meteorological variables such as wind. It must specify:

- conditioning feature names, not only numeric indices;
- direction convention;
- dynamic edge formula;
- normalization and self-loops;
- stability bounds and constants;
- whether the graph is directed;
- whether dynamic parameters are learned or fixed;
- configuration values for all meaningful hyperparameters.

Static and dynamic graph modes must be distinct, named, and independently recorded before results are compared.

## 7. Residual Learning Module

The residual formulation is:

$$r = y_{actual} - y_{baseline}$$

$$\hat{y} = y_{baseline} + \hat{r}$$

The baseline forecast must be an input available at prediction time. Residual targets must be constructed without future leakage. Evaluation must report reconstructed final predictions in original physical units, not residual metrics alone.

The baseline source, target order, residual scaler, and reconstruction rule must be part of the experiment manifest.

## 8. Prediction Head

The prediction head must produce the configured target count over the configured forecast horizon and site count:

- single-site: target outputs for the defined residual target horizon;
- multi-site: `(Tout, sites, targets)` outputs.

Output ordering must be recorded with the target schema. Any multi-horizon direct or autoregressive choice must be explicit in configuration.

## 9. Mask Handling

Masks must distinguish observed, imputed, engineered, and target-valid values where relevant. The contract must define:

- mask shape;
- valid-value convention;
- imputation interaction;
- scaler-fitting interaction;
- loss interaction;
- metric interaction;
- model-attention interaction.

The current legacy behavior remains documented separately. Future model implementations must not silently assume that carrying a mask means the model uses it.

## 10. Loss Function

The loss must be defined in the experiment configuration and report. Masked losses must use valid observations only. If Huber/SmoothL1 is used, its delta must be recorded. Target weighting, site weighting, horizon weighting, and pollutant weighting must be explicit.

Loss definitions do not replace evaluation definitions: final evaluation must use clearly specified original-unit metrics.

## 11. Training Loop Requirements

Every training run must record:

- optimizer and scheduler;
- learning rate and weight decay;
- batch size;
- epochs and early-stopping policy;
- gradient clipping;
- seed and deterministic-execution settings;
- device and package versions;
- training/validation histories;
- selected checkpoint rule;
- dataset, schema, scaler, and code identifiers.

Training changes are not part of Phase 6.

## 12. Evaluation Requirements

Future evaluation must:

- inverse-transform predictions and targets into original units;
- use valid observation masks;
- compute global masked MAE and RMSE rather than averaging batch RMSE scalars;
- report O3 and NO2 separately;
- report horizon-wise and site-wise results for multi-site models;
- state the aggregation scope;
- preserve legacy results separately from research-grade results;
- record evaluator version and configuration.

## 13. Configuration Requirements

The future configuration contract must include:

- dataset and raw-data identifiers/hashes;
- sites and site ordering;
- feature schema and target schema;
- `Tin`, `Tout`, stride;
- split policy, purge gap, and embargo;
- scaler identifiers;
- model family and architecture version;
- graph mode and graph parameters;
- hidden dimension, layers, heads, dropout;
- optimizer, learning rate, batch size, epochs;
- seed and deterministic settings;
- checkpoint and evaluation policy.

Source edits must not be the normal mechanism for changing experiment parameters.

## 14. Checkpoint Requirements

Future checkpoints must be accompanied by or contain metadata for:

- architecture version;
- feature schema version/hash;
- target schema/order;
- input dimensions;
- site count/order;
- `Tin` and `Tout`;
- graph mode;
- full configuration;
- training Git commit;
- dataset/scaler identifiers;
- environment/package versions;
- checkpoint hash.

Existing opaque checkpoints remain frozen historical artifacts.

## 15. Experiment Tracking Requirements

Each experiment must have a unique ID and a manifest containing:

- timestamp;
- code commit;
- raw-data and dataset hashes;
- feature-schema and scaler hashes;
- preprocessing version;
- complete configuration;
- environment package list;
- random seed;
- checkpoint hash;
- training, validation, and test metrics;
- warnings, deviations, and unresolved limitations.

The manifest must support independent identification of the exact inputs and code that produced a result.

## 16. Future API Boundary

The research pipeline owns preprocessing, training, evaluation, and artifacts. The inference service owns validated request handling, model loading, inference, and standardized responses. The frontend owns interaction and visualization.

The API must not become the source of truth for research preprocessing or metric definitions. Frontend code must not embed model or data transformation logic.

## 17. Implementation Boundary

This document does not authorize:

- model implementation;
- preprocessing changes;
- dataset regeneration;
- leakage-policy changes;
- retraining;
- checkpoint changes;
- API/UI changes;
- dependency changes.

Those activities begin only in explicitly approved later phases.
