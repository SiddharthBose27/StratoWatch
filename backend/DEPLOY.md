# StratoWatch Backend Deployment Guide

The StratoWatch backend is a **FastAPI + Python** application that requires:
- Python 3.12+
- PyTorch 2.x (for multi-site model checkpoints)
- XGBoost (for single-site model)
- scikit-learn (for scalers)
- NumPy, Pandas

Because of these requirements, **the backend cannot be deployed on Vercel** (Vercel only supports Node.js / static sites natively). The frontend is deployed on Vercel, and the backend must be deployed separately.

---

## Recommended: Railway

Railway supports Python natively and is the easiest path for this project.

### 1. Create a Railway project

1. Go to [railway.app](https://railway.app) and sign in.
2. Create a new project and connect your GitHub repo.
3. Set the root directory to `backend/` (not the repo root).

### 2. Add a `Procfile` (already provided in `backend/`)

```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 3. Set environment variables in Railway

| Variable               | Value                                    |
|------------------------|------------------------------------------|
| `PORT`                 | (set automatically by Railway)           |
| `CORS_ALLOWED_ORIGINS` | `https://your-vercel-app.vercel.app`    |

### 4. Set environment variable in Vercel

In your Vercel project dashboard:
- Go to **Settings > Environment Variables**
- Add `VITE_API_BASE_URL` = `https://your-railway-url.railway.app`
- Redeploy the Vercel frontend

---

## Alternative: Render

1. Create a new **Web Service** on [render.com](https://render.com).
2. Set root directory to `backend/`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Set `CORS_ALLOWED_ORIGINS` to your Vercel URL.

---

## Local Development

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The Vite dev proxy (`vite.config.ts`) forwards `/api` and `/health` to `127.0.0.1:8000` automatically during local development. No environment variables are needed locally.

---

## Important Notes

- The backend serves **frozen model evaluation artifacts** — it does not retrain models at runtime.
- Model artifacts (`.pt`, `.joblib`, `.npy`) are excluded from the git repository by `.gitignore`. They must be present in the deployment environment.
- The multi-site NPZ data file is also excluded. The frozen evaluation metrics are loaded from JSON files which ARE included in the repo.
