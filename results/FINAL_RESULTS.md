# StratoWatch — Final Results

**Single source of truth for all numerical results used in the research paper.**

All numbers in this file are read directly from frozen evaluation JSON artifacts.
No numbers are estimated, rounded early, or copied from memory.

Generated: 2026-09-12
Sources: `backend/stratowatch_single_site/outputs/` and `backend/stratowatch_multi_site/outputs/`

---

## Experiment Configuration

### Single-Site

| Parameter              | Value                         |
|------------------------|-------------------------------|
| Input features         | 134                           |
| Input window           | 24 hours                      |
| Forecast horizon       | 6 hours                       |
| Targets                | O3, NO2                       |
| Test samples           | 6,464                         |
| Evaluation units       | Near-normalized (StandardScaler mean≈0, scale≈1) |
| Random seed            | Not fixed for baselines; 42 for Transformer |
| Dataset artifact       | `stratowatch_single_v2_scaled.npz` (Phase 7 official test split) |

> **Important unit note:** The single-site target scaler has mean ≈ [−0.226, 0.162] and scale ≈ [1.082, 1.005]. This is a near-identity transform applied to already-normalized data. Single-site MAE/RMSE values are therefore in the normalized-input space, **not** raw µg/m³. They are not directly comparable with multi-site real-unit metrics without rescaling.

### Multi-Site

| Parameter              | Value                         |
|------------------------|-------------------------------|
| Input features         | 28 (per site)                 |
| Number of sites        | 7                             |
| Input window           | 24 hours                      |
| Forecast horizon       | 6 hours                       |
| Targets                | O3, NO2                       |
| Evaluation units       | Real units (µg/m³), inverse-scaled using target_scaler.json |
| Masking                | Global mask weighted          |
| Random seed            | 42                            |
| Dataset artifact       | `splits_final_Yscaled_Tin24_Tout6_stride1.npz` |
| Target scaler          | O3: mean=31.967 µg/m³, std=34.720; NO2: mean=39.133 µg/m³, std=29.909 |

---

## Single-Site Results

All single-site metrics are in **near-normalized units** (see note above).

### Overall Metrics

| Model           | MAE    | RMSE   | R²      |
|-----------------|--------|--------|---------|
| Forecast-only baseline | 1.2384 | 1.5116 | −3.0296 |
| Random Forest   | 0.4392 | 0.6526 | 0.2489  |
| **XGBoost**     | **0.4039** | **0.5906** | **0.3848** |
| TCN             | 0.5588 | 0.7427 | 0.0273  |
| LSTM            | 0.6616 | 0.8296 | −0.2138 |
| Transformer     | 0.5700 | 0.7524 | 0.0017  |

**Best model: XGBoost** (lowest MAE, RMSE, highest R²)

> Source: `backend/stratowatch_single_site/outputs/final_baselines/final_single_site_results.json`

---

### XGBoost — Per-Target Metrics

| Target | MAE    | RMSE   | R²      |
|--------|--------|--------|---------|
| O3     | 0.3045 | 0.4769 | 0.0609  |
| NO2    | 0.5034 | 0.6858 | 0.3561  |

> Source: `backend/stratowatch_single_site/outputs/final_baselines/xgboost/test_metrics.json`

---

### XGBoost — Horizon-wise Metrics

| Horizon | MAE    | RMSE   | R²      |
|---------|--------|--------|---------|
| H+1     | 0.3497 | 0.5157 | 0.5306  |
| H+2     | 0.3731 | 0.5469 | 0.4723  |
| H+3     | 0.3968 | 0.5810 | 0.4046  |
| H+4     | 0.4225 | 0.6145 | 0.3342  |
| H+5     | 0.4364 | 0.6303 | 0.2998  |
| H+6     | 0.4451 | 0.6448 | 0.2673  |

---

### Random Forest — Per-Target Metrics

| Target | MAE    | RMSE   | R²      |
|--------|--------|--------|---------|
| O3     | 0.3057 | 0.4816 | 0.0422  |
| NO2    | 0.5727 | 0.7873 | 0.1514  |

---

### TCN — Per-Target Metrics

| Target | MAE    | RMSE   | R²       |
|--------|--------|--------|----------|
| O3     | 0.5822 | 0.7365 | −1.2398  |
| NO2    | 0.5353 | 0.7488 | 0.2324   |

---

### LSTM — Per-Target Metrics

| Target | MAE    | RMSE   | R²       |
|--------|--------|--------|----------|
| O3     | 0.7544 | 0.8810 | −2.2047  |
| NO2    | 0.5687 | 0.7749 | 0.1779   |

---

### Transformer — Per-Target Metrics

| Target | MAE    | RMSE   | R²       |
|--------|--------|--------|----------|
| O3     | 0.5807 | 0.7392 | −1.2563  |
| NO2    | 0.5593 | 0.7654 | 0.1980   |

---

## Multi-Site Results

All multi-site metrics are in **real units (µg/m³)**.

### Overall Metrics

| Model                    | MAE     | RMSE    | R²     |
|--------------------------|---------|---------|--------|
| **ST-Transformer**       | **18.8349** | 27.9747 | 0.3437 |
| Static Graph-ST          | 19.0384 | 27.0781 | 0.3850 |
| Dynamic Wind Graph-ST    | 19.0464 | **26.9455** | **0.3911** |

> ST-Transformer achieves best MAE. Dynamic Wind Graph-ST achieves best RMSE and R².

---

### ST-Transformer — Per-Target Metrics (µg/m³)

| Target | MAE     | RMSE    | R²     |
|--------|---------|---------|--------|
| O3     | 19.0729 | 29.2425 | 0.4078 |
| NO2    | 18.5969 | 26.6466 | 0.2078 |

> Source: `backend/stratowatch_multi_site/outputs/final_evaluation/final_multisite_results.json`

Note: Horizon-wise metrics are not available for ST-Transformer.

---

### Static Graph-ST — Per-Target Metrics (µg/m³)

| Target | MAE     | RMSE    | R²     |
|--------|---------|---------|--------|
| O3     | 18.6939 | 27.5544 | 0.4742 |
| NO2    | 19.3830 | 26.5933 | 0.2110 |

### Static Graph-ST — Horizon-wise Metrics (µg/m³)

| Horizon | MAE     | RMSE    | R²     |
|---------|---------|---------|--------|
| H+1     | 18.9347 | 27.0340 | 0.3867 |
| H+2     | 18.9478 | 27.0556 | 0.3858 |
| H+3     | 18.9466 | 26.8322 | 0.3962 |
| H+4     | 19.0833 | 27.0781 | 0.3851 |
| H+5     | 19.1280 | 27.1005 | 0.3843 |
| H+6     | 19.1902 | 27.3657 | 0.3722 |

---

### Dynamic Wind Graph-ST — Per-Target Metrics (µg/m³)

| Target | MAE     | RMSE    | R²     |
|--------|---------|---------|--------|
| O3     | 18.7858 | 27.4837 | 0.4769 |
| NO2    | 19.3071 | 26.3963 | 0.2226 |

### Dynamic Wind Graph-ST — Horizon-wise Metrics (µg/m³)

| Horizon | MAE     | RMSE    | R²     |
|---------|---------|---------|--------|
| H+1     | 18.8561 | 26.7963 | 0.3975 |
| H+2     | 18.9590 | 26.8589 | 0.3947 |
| H+3     | 18.9966 | 26.7057 | 0.4018 |
| H+4     | 19.0893 | 26.9955 | 0.3889 |
| H+5     | 19.1414 | 27.0127 | 0.3882 |
| H+6     | 19.2365 | 27.2998 | 0.3752 |

---

## Model Configurations

### Single-Site XGBoost
- **Architecture**: XGBRegressor, 12 independent regressors (1 per horizon × target)
- **Target mode**: Direct target prediction (NOT residual learning)
- `residual_learning: false` (verified in metadata.json and evaluate_xgb.py)
- Input: 24h × 134 features → flattened to 3,216 features
- Output: 6 horizons × 2 targets (O3, NO2) predicted independently

### Single-Site Transformer (TemporalTransformer)
- Checkpoint: `outputs/final_single_site/best_transformer.pt`
- Device: MPS (Apple Silicon)
- Seed: 42

### Multi-Site ST-Transformer
- Model class: `STTransformer`
- Features: 28 per site, 7 sites, 24h input, 6h output
- Seed: 42
- Source: `src.eval_baseline_realunits`

### Multi-Site Static Graph-ST
- Model class: `StaticGraphSTTransformer`
- Checkpoint: `outputs/checkpoints/static_graph_st_best.pt`
- Device: MPS, Seed: 42

### Multi-Site Dynamic Wind Graph-ST
- Model class: `DynamicWindGraphSTTransformer`
- Checkpoint: `outputs/checkpoints/dynamic_wind_graph_st_best.pt`
- Device: MPS, Seed: 42

---

## Reproducibility Information

All evaluation metrics were computed in prior Phase 7 runs and stored as frozen JSON artifacts.
The API serves these frozen JSON files directly — no re-computation occurs at request time.

To reproduce the single-site results:
```bash
cd backend/stratowatch_single_site/baselines
python evaluate_xgb.py    # requires model files and data
python evaluate_rf.py
python evaluate_tcn.py
python evaluate_lstm.py
```

To reproduce the multi-site results:
```bash
cd backend/stratowatch_multi_site
python -m src.eval_baseline_realunits
python -m src.eval_graph_realunits --model static
python -m src.eval_graph_realunits --model dynamic
```

All scripts require the full data artifacts (`.npz`, `.pkl`, `.pt` files) which are excluded from git by `.gitignore`.

---

## Known Limitations

1. **Single-site unit mismatch**: Metrics are in near-normalized space, not raw µg/m³, due to the nearly-identity target scaler. Direct comparison with multi-site µg/m³ metrics is misleading.

2. **Spike underestimation**: All models underestimate extreme pollution spikes. This is a known limitation of mean-squared loss-trained models on heavy-tailed pollution distributions.

3. **O3 forecasting is harder**: O3 R² is consistently lower (and often negative for neural baselines) compared to NO2. O3 shows more non-linear seasonal dynamics.

4. **Multi-site horizon-wise metrics for ST-Transformer**: Not available — the ST-Transformer evaluation script does not compute horizon-wise breakdowns.

5. **Seven-site frozen checkpoints only**: The multi-site models are trained for exactly 7 Delhi monitoring stations. The application falls back to a persistence baseline for other site counts.

6. **Backend not deployed**: The Vercel-deployed frontend cannot execute forecasts without a separately-deployed backend. See `backend/DEPLOY.md` for instructions.
