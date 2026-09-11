# Phase 6 Decisions

Phase 6 is specification-only. These decisions do not implement or alter the runtime pipeline.

## DEC-001 - Preserve the 179-feature branch as historical reference

- **Decision:** The 179-feature single-site branch remains frozen as a historical reference baseline.
- **Evidence:** Byte-identical active/full/tuned 179-wide sequence artifacts and matching 179-input checkpoints; exact 45-feature schema remains unresolved.
- **Rationale:** Historical numerical results must remain available without inventing missing feature identities.
- **Consequence:** The 179 branch is not the source of truth for new feature engineering or reproducible preprocessing.
- **Implementation phase:** Existing freeze; enforce in future experiment tooling.

## DEC-002 - Adopt the 134-feature branch as the official reproducible single-site baseline

- **Decision:** The current notebook/no-satellite lineage is the official reproducible single-site baseline.
- **Evidence:** Current and adjacent notebooks are byte-identical and produce `X_seq=(43489,24,134)` with 134 persisted names.
- **Rationale:** This is the only single-site preprocessing lineage reproducible from available source.
- **Consequence:** New single-site research must use a versioned successor of this lineage or explicitly declare a new schema.
- **Implementation phase:** Phase 7.

## DEC-003 - Version every future feature schema

- **Decision:** Feature names, order, count, transformations, masks, units, and scaling are part of the model contract and require a schema version/hash.
- **Evidence:** The unresolved 134-versus-179 mismatch demonstrates that dimensions without metadata are not reproducible.
- **Rationale:** A checkpoint cannot be scientifically reproduced without its input schema.
- **Consequence:** Future checkpoints and manifests must reference an immutable feature schema.
- **Implementation phase:** Phase 7 and checkpoint tooling phases.

## DEC-004 - Use leakage-controlled temporal splitting for future experiments

- **Decision:** Future datasets must use a purge/embargo policy derived from window geometry.
- **Evidence:** The legacy stride-1 chronological split shares context across boundaries.
- **Rationale:** Future research comparisons must prevent train samples from sharing input/forecast timestamps with validation/test samples.
- **Consequence:** The current split remains legacy evidence; future split artifacts receive a new version and experiment ID.
- **Implementation phase:** Phase 8.

## DEC-005 - Fit future normalization statistics on training data only

- **Decision:** Learned feature and target normalization statistics may use training observations only, respecting masks.
- **Evidence:** The final multi-site pipeline records train-only scaler fitting; this is the required future rule.
- **Rationale:** Validation/test information must not influence model preparation.
- **Consequence:** Scaler scope and mask policy must be recorded in every experiment manifest.
- **Implementation phase:** Phase 7.

## DEC-006 - Require provenance metadata for future checkpoints

- **Decision:** Future checkpoints must contain or reference architecture, schema, target, site, geometry, graph, dataset, scaler, commit, configuration, and environment metadata.
- **Evidence:** Existing state-dict-only checkpoints establish dimensions but not feature semantics.
- **Rationale:** Tensor shapes alone do not establish scientific reproducibility.
- **Consequence:** Existing checkpoints remain frozen; new checkpoint format/version is required later.
- **Implementation phase:** Phase 9 and subsequent model phases.

## DEC-007 - Make graph and site ordering configuration-driven

- **Decision:** Future graph models must obtain site order, coordinates, adjacency, wind feature identity, and graph parameters from explicit versioned configuration/schema metadata.
- **Evidence:** Current graph code contains hardcoded site and wind-index assumptions, and static/dynamic paths are not fully distinct.
- **Rationale:** Spatial tensor position cannot be a hidden contract.
- **Consequence:** New graph experiments require explicit site and graph manifests.
- **Implementation phase:** Phase 10/11.

## DEC-008 - Use explicit global masked metric aggregation for future evaluation

- **Decision:** Future official MAE/RMSE must be computed from global valid-observation sums, not an unweighted average of batch-level scalar metrics.
- **Evidence:** Current multi-site evaluators average batch-level masked scalar metrics.
- **Rationale:** Final reporting must have a clearly defined aggregation independent of batch partitioning.
- **Consequence:** Legacy evaluator outputs remain historical and must be labeled as such; future evaluator changes require a new version.
- **Implementation phase:** Phase 12.

## DEC-009 - Keep research and product layers separate

- **Decision:** Research preprocessing/training/evaluation remain separate from inference service and frontend presentation.
- **Evidence:** Current API runners execute shared research artifacts synchronously and the frontend consumes model outputs.
- **Rationale:** Scientific contracts should not be hidden in product wiring.
- **Consequence:** Future API hardening cannot silently redefine research data or metrics.
- **Implementation phase:** Phase 13/14.

## DEC-010 - Do not claim historical schema resolution

- **Decision:** The historical 179-feature schema is formally frozen as partially documented and unresolved.
- **Evidence:** Phase 5 found no names/order in Git, notebooks, backups, caches, checkpoints, environments, or scoped project remnants; numeric matches do not establish identity.
- **Rationale:** Correlation or duplicate structure is not historical provenance.
- **Consequence:** No synthetic 179-feature schema may be created without new direct evidence.
- **Implementation phase:** Immediate policy; review only if new external evidence is supplied.
