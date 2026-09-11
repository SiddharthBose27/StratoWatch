# Phase 6 Report

## Objective

Phase 6 converted the accepted Phase 0-5 findings into a formal baseline freeze and research architecture specification before any StratoWatch 2.0 implementation work.

## Completed

Created:

- `docs/STRATOWATCH_RESEARCH_BASELINE.md`
- `docs/STRATOWATCH_2_ARCHITECTURE_SPEC.md`
- `docs/PHASE_6_DECISIONS.md`
- `docs/PHASE_6_REPORT.md`

The documents define the official baselines, data/schema contract, missing-data and normalization policies, leakage policy, evaluation contract, experiment manifest requirements, checkpoint requirements, research questions, ablations, scientific integrity policy, layer boundaries, and future roadmap.

Existing Phase 1-5 reports were preserved.

## Baseline Decision

**Official reproducible baseline: current documented 134-feature single-site pipeline + verified 28-feature multi-site pipeline.**

**Historical reference: 179-feature single-site lineage.**

The 179-feature branch is valid historical artifact lineage but is not fully reproducible because the identities and ordering of 45 dimensions remain unknown. It must not be used as a source of truth for new feature engineering.

## Scientific Policy

Future experiments must:

- version feature schemas and site mappings;
- fit normalization statistics using training data only;
- distinguish legacy stride-1 splits from leakage-controlled research-grade splits;
- derive purge/embargo separation from window geometry;
- compute future official masked metrics from global valid-observation sums;
- report original-unit O3 and NO2 results separately;
- record complete data, code, configuration, environment, seed, scaler, and checkpoint provenance;
- preserve historical results without silently changing their evaluation contract.

## Architecture

The planned StratoWatch 2.0 architecture is specified at the design level:

- Transformer temporal representation over hourly histories;
- configurable spatial graph/message passing or spatial attention;
- separately defined static and wind-conditioned dynamic graph modes;
- residual learning around an explicitly available baseline forecast;
- mask-aware training/evaluation contracts;
- configuration-driven site, feature, graph, and training parameters;
- metadata-bearing checkpoints;
- separate research pipeline, inference service, and frontend layers.

No architecture was implemented.

## Important Unresolved Items

- Exact 179-feature historical schema and ordering.
- Exact historical preprocessing source for the 45 unnamed dimensions.
- Whether any future external evidence can recover the missing schema.
- Existing technical debt documented in prior phase reports remains unfixed.

## Changes Made

Documentation/specification files only:

- `docs/STRATOWATCH_RESEARCH_BASELINE.md`
- `docs/STRATOWATCH_2_ARCHITECTURE_SPEC.md`
- `docs/PHASE_6_DECISIONS.md`
- `docs/PHASE_6_REPORT.md`

## Changes NOT Made

- No model implementation.
- No training or retraining.
- No data regeneration.
- No preprocessing changes.
- No feature/schema changes.
- No checkpoint or scaler changes.
- No API/UI changes.
- No dependency or package changes.
- No frontend/backend runtime changes.

## Next Phase

**Phase 7 — Research-Grade Data Contract + Reproducible Preprocessing Implementation**

Phase 7 is not started automatically.
