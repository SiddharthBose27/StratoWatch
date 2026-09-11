# Multi-site Raw Data Audit

**Source directory:** `backend/stratowatch_multi_site/data/raw/data/`

## Inventory

- 14 real CSV files: seven `_train_data.csv` files and seven `_unseen_input_data.csv` files.
- Site IDs are 1 through 7 in both groups.
- macOS `._` files also exist elsewhere under the raw tree; scripts 02 and 03 explicitly ignore them.
- `lat_lon_sites.txt` contains seven coordinate rows in site-ID order.

## Schemas

Training files all use this 16-column order:

`year, month, day, hour, O3_forecast, NO2_forecast, T_forecast, q_forecast, u_forecast, v_forecast, w_forecast, NO2_satellite, HCHO_satellite, ratio_satellite, O3_target, NO2_target`

Unseen files all use the same 14 input columns without `O3_target` and `NO2_target`. No schema difference was observed among train files or among unseen files.

## Observed counts

| Site      |  Train rows | Unseen rows |
| --------- | ----------: | ----------: |
| 1         |      25,081 |      10,872 |
| 2         |      25,969 |      10,920 |
| 3         |      21,913 |       9,288 |
| 4         |      24,505 |      10,488 |
| 5         |      25,081 |      10,848 |
| 6         |      26,353 |      11,160 |
| 7         |      22,777 |       9,720 |
| **Total** | **171,679** |  **73,296** |

Aggregate missing cells are 497,875 of 2,746,864 train cells (18.1252%) and 212,327 of 1,026,144 unseen cells (20.6917%). Missingness is concentrated in `ratio_satellite`, `NO2_satellite`, and `HCHO_satellite`; time and forecast columns were complete in the inspected files.

## Datetime and sampling

- Datetime is represented by four integer columns, not a timezone-aware datetime column.
- Constructed timestamps have no parse failures and no duplicate timestamps across the 14 files.
- Overall observed date range is 2019-07-10 00:00:00 through 2024-06-30 00:00:00.
- Rows are not continuous hourly sequences: one-hour gaps are common, but 25-, 49-, 73-hour and larger gaps also occur. The alignment script sorts and reindexes onto a union hourly timeline.
- No timezone field was present; timezone semantics are UNKNOWN.

## Coordinates

`backend/stratowatch_multi_site/data/raw/data/lat_lon_sites.txt` contains:

| Site | Latitude | Longitude |
| ---- | -------: | --------: |
| 1    | 28.69536 |  77.18168 |
| 2    | 28.57180 |  77.07125 |
| 3    | 28.58278 |  77.23441 |
| 4    | 28.82286 |  77.10197 |
| 5    | 28.53077 |  77.27123 |
| 6    | 28.72954 |  77.09601 |
| 7    | 28.71052 |  77.24951 |

## Audit limits

The statistics above were read directly from the CSVs. The repository does not store raw-data checksums, source provenance, or timezone declarations. The audit did not rewrite or normalize any raw file.
