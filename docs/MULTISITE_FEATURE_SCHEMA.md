# Multi-site Feature Schema

The active final feature order is persisted in `final_feature_scaler.json`, the final NPZ `feature_cols`, and the ST checkpoint metadata. The table uses one-based indices for readability.

| Index | Feature                | Source              | Transformation                                          | Scaled?                  | Used by model? | Confidence |
| ----: | ---------------------- | ------------------- | ------------------------------------------------------- | ------------------------ | -------------- | ---------- |
|     1 | year                   | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|     2 | month                  | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|     3 | day                    | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|     4 | hour                   | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|     5 | O3_forecast            | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|     6 | NO2_forecast           | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|     7 | T_forecast             | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|     8 | q_forecast             | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|     9 | u_forecast             | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|    10 | v_forecast             | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|    11 | w_forecast             | raw site CSV        | retained                                                | global 14-feature stage  | yes            | CONFIRMED  |
|    12 | NO2_satellite          | raw site CSV        | retained with mask                                      | global 14-feature stage  | yes            | CONFIRMED  |
|    13 | HCHO_satellite         | raw site CSV        | retained with mask                                      | global 14-feature stage  | yes            | CONFIRMED  |
|    14 | ratio_satellite        | raw site CSV        | retained with mask                                      | global 14-feature stage  | yes            | CONFIRMED  |
|    15 | sin_hour               | notebook cell 6     | cyclical encoding                                       | no final partial scaling | yes            | CONFIRMED  |
|    16 | cos_hour               | notebook cell 6     | cyclical encoding                                       | no final partial scaling | yes            | CONFIRMED  |
|    17 | sin_doy                | notebook cell 6     | day-of-year cyclical encoding                           | no final partial scaling | yes            | CONFIRMED  |
|    18 | cos_doy                | notebook cell 6     | day-of-year cyclical encoding                           | no final partial scaling | yes            | CONFIRMED  |
|    19 | sin_wday               | notebook cell 6     | weekday cyclical encoding                               | no final partial scaling | yes            | CONFIRMED  |
|    20 | cos_wday               | notebook cell 6     | weekday cyclical encoding                               | no final partial scaling | yes            | CONFIRMED  |
|    21 | NO2_satellite_is_obs   | notebook cell 7     | mask-derived 0/1 flag                                   | no final partial scaling | yes            | CONFIRMED  |
|    22 | NO2_satellite_age      | notebook cell 7     | hours since last observation, capped at Tin when unseen | final partial scaler     | yes            | CONFIRMED  |
|    23 | HCHO_satellite_is_obs  | notebook cell 7     | mask-derived 0/1 flag                                   | no final partial scaling | yes            | CONFIRMED  |
|    24 | HCHO_satellite_age     | notebook cell 7     | hours since last observation, capped at Tin when unseen | final partial scaler     | yes            | CONFIRMED  |
|    25 | ratio_satellite_is_obs | notebook cell 7     | mask-derived 0/1 flag                                   | no final partial scaling | yes            | CONFIRMED  |
|    26 | ratio_satellite_age    | notebook cell 7     | hours since last observation, capped at Tin when unseen | final partial scaler     | yes            | CONFIRMED  |
|    27 | site_lat               | `lat_lon_sites.txt` | broadcast across N/Tin/site axis                        | final partial scaler     | yes            | CONFIRMED  |
|    28 | site_lon               | `lat_lon_sites.txt` | broadcast across N/Tin/site axis                        | final partial scaler     | yes            | CONFIRMED  |

The first 14 features are the non-target columns preserved by `03_align_multisite.py`. The final partial scaler stores zero mean/one standard deviation for passthrough features and fitted values for the five listed scaled feature groups. The final NPZ and checkpoint both confirm width 28.

`X_mask` is carried alongside the feature tensor. Engineered observation/age and coordinate features receive all-one masks in the notebook. The current models accept masks through the dataset path but do not use `X_mask` in attention or graph propagation.

## Unknowns

- The exact raw-to-final artifact provenance is not hash-linked.
- The current final NPZ values were not recomputed from raw data, by design.
- The 14-feature global scaler is an intermediate-stage artifact by code lineage, but its creation run and exact relationship to current final values are not independently recorded.
