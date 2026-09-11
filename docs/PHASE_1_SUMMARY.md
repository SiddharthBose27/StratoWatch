# Phase 1 Summary

| Area              | Status             | Evidence                                                                                                                                                                                                   | Action                                                              |
| ----------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Dataset contract  | PARTIALLY VERIFIED | Single-site NPZ schema and active multi-site NPZ shapes were read; single-site feature metadata is 134 versus sequence width 179, and multi-site feature count is 28 rather than the Phase 0-described 14. | Preserve artifacts; audit schema/scaler provenance later.           |
| Single-site model | PARTIALLY VERIFIED | Transformer imports, constructor instantiates, and `best_model.pt` loads; residual reconstruction is present; the Joblib scaler loads with a version warning.                                              | Do not regenerate or replace scaler in Phase 1.                     |
| Multi-site model  | VERIFIED           | ST and Graph ST constructors import, checkpoints load, and synthetic forward shapes are `(B,6,7,2)`.                                                                                                       | Keep current dimensions and checkpoints frozen.                     |
| Graph model       | PARTIALLY VERIFIED | Adjacency/coordinates exist; wind indices are 8/9 and beta is 0.5; both UI graph modes route to the same evaluator; dynamic checkpoint absent.                                                             | Document distinction as unverified; do not redesign.                |
| Metrics           | VERIFIED           | Synthetic masked MAE/MSE/RMSE/Huber tests match current formulas; evaluator source confirms batch-scalar averaging.                                                                                        | Preserve current definitions.                                       |
| API               | VERIFIED           | Route table, health response, endpoint parameter names, and response keys were inspected.                                                                                                                  | Preserve route and field names.                                     |
| Frontend          | PARTIALLY VERIFIED | `package.json` and `package-lock.json` are present; build passes after locked dependency installation, while lint reports five existing `no-explicit-any` errors.                                          | Keep UI frozen; address lint only in a later audited phase.         |
| Dependencies      | PARTIALLY VERIFIED | Overlapping Python manifests have broad/unpinned versions; the single-site requirements filename is misspelled; backend manifest omits imported `joblib`/`seaborn`; npm audit reports 13 vulnerabilities.  | Document only; do not upgrade, fix, or consolidate now.             |
| Reproducibility   | PARTIALLY VERIFIED | Runtime artifacts exist locally and are recorded in `configs/baseline_manifest.json`; artifacts are ignored by Git, normalization is notebook-dependent, and one scaler is invalid.                        | Archive/provision artifacts and audit provenance in the next phase. |

## Overall Status

**PARTIAL.** The core multi-site dataset/model/API/metric contracts are measurable and verified locally. The baseline is not fully reproducible because the single-site feature schema remains unresolved, the final multi-site schema differs from Phase 0 documentation, and frontend checks require unavailable local dependencies.

## Frozen Findings

- Random Forest remains broken at the existing dictionary-key contract and was not fixed.
- Multi-site uploaded files remain ignored by the existing runtime and were not connected.
- Static/dynamic graph behavior remains ambiguous and was not redesigned.
- No models were retrained, no datasets regenerated, and no ignored artifacts were modified.
