# Artifact Integrity Report

This report records read-only existence, size, loadability, and schema checks performed during Phase 2. Binary artifacts were loaded only for inspection; none were rewritten.

| Path                                                                                         | Exists | Size / shape                                                 | Readable/loadable                         | Status                             |
| -------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------ | ----------------------------------------- | ---------------------------------- |
| `backend/stratowatch_single_site/data/sequences.npz`                                         | yes    | 708,945,801 bytes; `X_seq=(43489,24,179)`, `y_seq=(43489,2)` | yes                                       | PARTIALLY VERIFIED                 |
| `backend/stratowatch_single_site/data/feature_list.json`                                     | yes    | 2,927 bytes; 134 names                                       | yes                                       | MISMATCH: width 134 vs 179         |
| `backend/stratowatch_single_site/data/y_res_scaler.pkl`                                      | yes    | 631 bytes                                                    | yes with `joblib.load()`; version warning | VERIFIED WITH VERSION WARNING      |
| `backend/stratowatch_single_site/outputs/checkpoints/best_model.pt`                          | yes    | 1,699,125 bytes                                              | yes; state dict loads                     | VERIFIED                           |
| `backend/stratowatch_multi_site/data/raw/data/*.csv`                                         | yes    | 14 files; 7 train/7 unseen                                   | yes; schemas/date parsing checked         | VERIFIED                           |
| `backend/stratowatch_multi_site/data/processed/splits_final_Yscaled_Tin24_Tout6_stride1.npz` | yes    | 16,964,354 bytes; X width 28/Y width 2                       | yes; keys/shapes checked                  | VERIFIED                           |
| `backend/stratowatch_multi_site/configs/global_scaler.json`                                  | yes    | 997 bytes; 14 features                                       | yes                                       | INTERMEDIATE/STALE FOR FINAL INPUT |
| `backend/stratowatch_multi_site/configs/final_feature_scaler.json`                           | yes    | 1,425 bytes; 28 features                                     | yes                                       | VERIFIED METADATA                  |
| `backend/stratowatch_multi_site/configs/target_scaler.json`                                  | yes    | 179 bytes; 2 targets                                         | yes                                       | VERIFIED                           |
| `backend/stratowatch_multi_site/data/processed/adjacency_final.npy`                          | yes    | 520 bytes; `(7,7)`                                           | yes                                       | VERIFIED                           |
| `backend/stratowatch_multi_site/configs/site_coords_raw.npy`                                 | yes    | 184 bytes; `(7,2)`                                           | yes                                       | VERIFIED                           |
| `backend/stratowatch_multi_site/outputs/checkpoints/st_transformer_best.pt`                  | yes    | 9,547,131 bytes                                              | yes; payload and state load               | VERIFIED                           |
| `backend/stratowatch_multi_site/outputs/checkpoints/graph_st_best.pt`                        | yes    | 9,596,603 bytes                                              | yes; payload and state load               | VERIFIED                           |

## Integrity limitations

- No hashes were recorded for the large binary artifacts.
- No source-to-artifact provenance manifest exists beyond the Phase 2 observations.
- “Readable” means loadable by the current local Python environment; it does not prove cross-version compatibility.
- The report deliberately does not compare regenerated numerical contents because regeneration is prohibited in this phase.
