# StratoWatch
A forecasting + evaluation hub for air-quality intelligence: single-site residual correction and multi-site spatio-temporal learning for O₃ and NO₂ — with metrics, plots, and dashboards.

> A full-stack spatio-temporal air‑quality forecasting system that blends transformer‑based modeling with graph‑aware spatial reasoning and a web UI for fast evaluation.

![React](https://img.shields.io/badge/React-19.x-61dafb?logo=react&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-7.x-646CFF?logo=vite&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-1.x-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-1a6aff)

---

## 🚀 Features
- **Single‑site residual learning** for O₃/NO₂ using XGBoost, LSTM, TCN, and transformer baselines
- **Multi‑site spatio‑temporal forecasting** with transformer + graph variants
- **FastAPI inference API** with file upload support
- **Metrics + plots** generation and UI rendering
- **Web UI** built with React + Vite for interactive runs

---

## 🏗 Architecture Overview
**Frontend**
- React + Vite UI for uploads, model selection, and results

**Backend**
- FastAPI server
- Runs single‑site and multi‑site pipelines
- Returns metrics and base64 plots

**Database**
- None (local file + preprocessed NPZ)

**AI / ML Components**
- Single‑site residual learning pipeline
- Multi‑site graph spatio‑temporal transformer pipeline

**Authentication**
- None (local dev API)

**ASCII Diagram**
```
+-------------+       +-------------------+       +-------------------------+
|  Frontend   |  ---> |   FastAPI Backend  |  ---> |   Model Runners / ML     |
|  React/Vite |       |  /api/* endpoints  |       |  Single & Multi‑site     |
+-------------+       +-------------------+       +-------------------------+
        |                          |                             |
        |                          |                             |
        |                    Metrics + Plots <-------------------+
        +-------------------- UI Rendering ----------------------+
```

---

## 🛠 Tech Stack

### Frontend
- **Framework**: React 19
- **Routing**: react‑router‑dom
- **Styling**: Tailwind CSS (PostCSS)
- **Charts**: Recharts
- **Animation**: Framer Motion

### Backend
- **Runtime**: Python 3.12+ (local env uses 3.13)
- **Framework**: FastAPI
- **ML**: PyTorch, XGBoost, scikit‑learn
- **Data**: NumPy, Pandas

### DevOps / Tools
- **Bundler**: Vite
- **Linting**: ESLint
- **Testing**: Not configured

---

## 📂 Project Structure
```
stratowatch/
├── backend/                     # FastAPI entrypoint + model runners
│   ├── app/
│   ├── stratowatch_single_site/
│   └── stratowatch_multi_site/
├── public/
│   └── assets/plots/             # UI images
├── src/
│   ├── components/
│   └── pages/                    # Home, SingleSite, MultiSite, Methodology, Docs
├── package.json
├── vite.config.ts
├── requirement.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- **Node**: 18+
- **Python**: 3.12+ recommended
- **Torch**: Required for multi‑site plot generation
- **XGBoost**: Required for single‑site residual model

### Steps
```bash
git clone <your-repo-url>
cd stratowatch

# Frontend
npm install
npm run dev
```

```bash
# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r /Users/siddharthbose/Desktop/Projects/stratowatch/requirement.txt
python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 🔐 Environment Variables
| Variable | Description | Required |
|---|---|---|
| None | No environment variables required for local run | No |

---

## 📸 Screenshots
> Example plots used by the UI

- O₃ True vs Pred: `public/assets/plots/o3_true_vs_pred.png`
- O₃ Residual Distribution: `public/assets/plots/o3_residual_distribution.png`
- O₃ Confusion Matrix: `public/assets/plots/XGB_O3_cm.png`
- O₃ Scatter: `public/assets/plots/o3_scatter.png`

---

## 🧠 How It Works

### Single‑Site Pipeline
- Learns a **residual correction**: `y = forecast + Δ`
- Baselines include Random Forest, XGBoost, LSTM, TCN, and transformer
- Produces MAE/RMSE/R² + confusion matrices

### Multi‑Site Pipeline
- Trains for **7 monitoring stations** with 24‑hour history and 6‑hour horizon
- Variants include:
  - ST Transformer
  - Static Graph‑ST
  - Dynamic Wind Graph‑ST
- Generates horizon‑wise metrics, residual plots, and embedding analysis

### System Flow
1. User uploads CSVs via UI
2. FastAPI stores files and triggers pipeline
3. Runner returns metrics + plots
4. UI renders cards + figures

---

## 📈 Future Improvements
- Stronger wind‑aware adjacency modeling
- Adaptive graph learning for spatial structure
- Uncertainty estimation for extreme pollution spikes
- Better per‑site calibration

---

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push branch: `git push origin feature/your-feature`
5. Open a pull request

---

## 📝 License
License not specified yet.

---

## ⭐ Why This Project Matters
StratoWatch brings together robust time‑series modeling and spatial reasoning in a practical, end‑to‑end system—making advanced air‑quality forecasting accessible through a clean, developer‑friendly interface.
