# StratoWatch

Air-quality forecasting system for Delhi — O₃ and NO₂, 6-hour horizon, single-site and multi-site spatio-temporal models.

[![React](https://img.shields.io/badge/React-19.x-61dafb?logo=react&logoColor=white)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-7.x-646CFF?logo=vite&logoColor=white)](https://vitejs.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-1a6aff)](https://xgboost.readthedocs.io)

---

## Problem

Delhi experiences some of the most severe air-quality episodes globally. Accurate short-horizon forecasting of ground-level O₃ and NO₂ is challenging because:

- Both pollutants have non-linear photochemical dynamics
- Meteorological inputs (temperature, humidity, wind) interact with emissions in complex ways
- Spatial autocorrelation across monitoring stations can improve forecasts but adds architectural complexity
- Short-horizon (1–6 hour) targets are sensitive to local micro-dynamics

StratoWatch provides a practical, end-to-end forecasting system that explores single-site and multi-site approaches and compares them under a unified evaluation protocol.

---

## Final System

### Single-Site Forecasting

- Input: 24-hour history × 134 meteorological and pollutant features
- Output: 6-hour-ahead forecasts for O₃ and NO₂
- Evaluation: Phase 7 official test split (6,464 windows)
- Baselines compared: XGBoost, Random Forest, TCN, LSTM, TemporalTransformer

**The final deployed single-site model is XGBoost — a direct target prediction model.** It uses 12 independent regressors (one per forecast horizon × target), each trained on 24h × 134 flattened features. This is not a residual-learning pipeline.

### Multi-Site Forecasting

- Input: 24-hour history × 28 features per site × 7 Delhi monitoring stations
- Output: 6-hour-ahead forecasts for O₃ and NO₂ at all 7 stations simultaneously
- Architecture variants:
  - **ST-Transformer**: Baseline spatio-temporal transformer
  - **Static Graph-ST**: Graph-augmented transformer with fixed adjacency (distance-based)
  - **Dynamic Wind Graph-ST**: Graph-augmented transformer with wind-aware dynamic adjacency

---

## Models

### Single-Site (frozen)

| Model           | Type                      | Status  |
|-----------------|---------------------------|---------|
| XGBoost         | Direct target XGBRegressor | Frozen  |
| Random Forest   | Direct target RF           | Frozen  |
| TCN             | Temporal Conv. Network     | Frozen  |
| LSTM            | Recurrent baseline         | Frozen  |
| Transformer     | TemporalTransformer        | Frozen  |

### Multi-Site (frozen, 7-station Delhi)

| Model                 | Type                              | Status  |
|-----------------------|-----------------------------------|---------|
| ST-Transformer        | Spatio-temporal transformer       | Frozen  |
| Static Graph-ST       | Static-graph ST-Transformer       | Frozen  |
| Dynamic Wind Graph-ST | Wind-aware dynamic-graph ST-Transformer | Frozen |

---

## Results

> **Unit note**: Single-site metrics are in near-normalized space (target scaler is near-identity). Multi-site metrics are in real units (µg/m³). They are not directly comparable as-is.

See [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md) for the complete verified metrics table.

### Single-Site (near-normalized units)

| Model         | MAE    | RMSE   | R²     |
|---------------|--------|--------|--------|
| XGBoost       | 0.4039 | 0.5906 | 0.3848 |
| Random Forest | 0.4392 | 0.6526 | 0.2489 |
| TCN           | 0.5588 | 0.7427 | 0.0273 |
| Transformer   | 0.5700 | 0.7524 | 0.0017 |
| LSTM          | 0.6616 | 0.8296 | −0.214 |

### Multi-Site (µg/m³)

| Model                 | MAE    | RMSE   | R²     |
|-----------------------|--------|--------|--------|
| ST-Transformer        | 18.835 | 27.975 | 0.344  |
| Static Graph-ST       | 19.038 | 27.078 | 0.385  |
| Dynamic Wind Graph-ST | 19.046 | 26.945 | 0.391  |

---

## Deployment

**Frontend**: Deployed on Vercel (static React + Vite build).

**Backend**: The FastAPI backend requires Python, PyTorch, XGBoost, and model artifacts. It **cannot** run on Vercel and must be deployed separately (Railway, Render, or similar).

See [`backend/DEPLOY.md`](backend/DEPLOY.md) for step-by-step deployment instructions.

To connect a deployed backend to the Vercel frontend, set the `VITE_API_BASE_URL` environment variable in the Vercel project dashboard.

---

## Research

The research paper associated with this project is located in `docs/` (not yet publicly linked).

The complete verified experimental results are in [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md).

---

## Limitations

- **Spike underestimation**: All models underestimate extreme pollution spikes, a known limitation of MSE-trained models on heavy-tailed distributions.
- **O₃ is harder to forecast**: O₃ R² is consistently lower than NO₂ across all models, reflecting the more complex photochemical dynamics of ozone.
- **Single-site metric units**: Metrics are in near-normalized space — this is a limitation of the Phase 7 scaler artifact that was fit on already-normalized data.
- **7-site restriction**: Multi-site models are trained for exactly 7 Delhi stations. Other configurations fall back to persistence baseline.
- **Backend not cloud-deployed**: The current production Vercel deployment shows a connectivity error until the backend is deployed to a cloud service.

---

## Tech Stack

### Frontend
- React 19, react-router-dom, Vite 7
- Tailwind CSS (PostCSS)
- Recharts, Framer Motion

### Backend
- Python 3.12+, FastAPI
- PyTorch 2.x, XGBoost 2.x, scikit-learn
- NumPy, Pandas, joblib

### Deployment
- Frontend: Vercel
- Backend: Railway / Render (requires manual setup — see `backend/DEPLOY.md`)

---

## Project Structure

```
stratowatch/
├── backend/
│   ├── app/                          # FastAPI app + model runners
│   │   ├── main.py                   # API endpoints, CORS config
│   │   ├── runner.py                 # Single-site evaluation runner
│   │   └── multi_runner.py           # Multi-site evaluation runner
│   ├── stratowatch_single_site/      # Single-site training code + artifacts
│   │   ├── baselines/                # XGBoost, RF, TCN, LSTM train/eval scripts
│   │   ├── models/                   # TemporalTransformer model definition
│   │   └── outputs/                  # Frozen evaluation artifacts (JSON, npy, plots)
│   ├── stratowatch_multi_site/       # Multi-site training code + artifacts
│   │   ├── src/                      # Model definitions + eval scripts
│   │   ├── configs/                  # Target scaler + feature schemas
│   │   └── outputs/                  # Frozen checkpoints + evaluation artifacts
│   ├── DEPLOY.md                     # Backend deployment guide
│   └── Procfile                      # For Railway deployment
├── src/
│   ├── components/                   # Navbar, shared UI
│   └── pages/                        # Home, SingleSite, MultiSite, Methodology, Docs
├── results/
│   └── FINAL_RESULTS.md              # Verified final metrics (single source of truth)
├── docs/                             # Phase reports, architecture docs
├── vercel.json                       # Vercel SPA routing config
├── .env.example                      # Environment variable template
├── vite.config.ts
└── package.json
```

---

## Local Setup

### Frontend

```bash
npm install
npm run dev
# Visit http://localhost:5173
```

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The Vite dev server proxies `/api` and `/health` to the local backend automatically.
