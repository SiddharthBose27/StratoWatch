# 🌬️ Luftwächter: Dynamic Wind-Aware Graph Spatio-Temporal Transformer for Multi-Site Air Quality Forecasting

# 📌 Overview

Luftwächter is a multi-site air quality forecasting system built using a Graph-Enhanced Spatio-Temporal Transformer architecture.

The model predicts:
O₃ (Ozone)
NO₂ (Nitrogen Dioxide)

Across:
7 monitoring stations
6-hour multi-horizon forecasting
Using 24-hour historical context

The project explores whether incorporating spatial graph structure and dynamic wind-aware adjacency improves forecasting performance over a pure transformer baseline.

The model integrates:
Temporal attention (per-site history modeling)
Graph convolution (spatial propagation)
Dynamic wind-aware adjacency weighting
Multi-horizon forecasting (6-hour ahead)
Multi-target prediction (O3, NO2)

# 🏗 Architecture

The system is composed of:
Input Projection
Projects raw features → d_model = 128
Temporal Transformer Encoder
Self-attention across time (per site)
Graph Convolution Layer
Spatial propagation using adjacency matrix
Static graph
Dynamic wind-weighted graph
Spatial Transformer Encoder
Attention across sites
Direct Multi-Horizon Forecast Head
Predicts all 6 future steps simultaneously

📊 Model Variants

Model	                Description
ST Transformer	        Temporal + Spatial attention (no graph)
Graph-ST	            Adds spatial propagation via fixed adjacency
Dynamic Wind Graph-ST	Adjacency dynamically modulated by wind direction

# 📊 Performance Table 

Based on your results:

ST Transformer
MAE: 18.69
RMSE: 26.66

Static Graph-ST
MAE: 19.01
RMSE: 26.99

Dynamic Wind Graph-ST
MAE: 19.00
RMSE: 26.99

(Your latest dynamic result: 19.0047 MAE, 26.99 RMSE)


# 🔎 Key Findings

1️⃣ Temporal modeling dominates performance
The baseline ST Transformer achieved the best overall MAE.

2️⃣ Graph models learn meaningful spatial embeddings
PCA projection of learned embeddings shows clear site clustering.

3️⃣ Wind-aware adjacency did not significantly improve aggregate MAE
Possible reasons:
Wind information already encoded in input features
Spatial correlations weaker than expected
Temporal signal dominates predictive power

4️⃣ Model underestimates extreme pollution spikes
Residual analysis shows:
Mean residual ≈ -7.6
Heavy negative tail
Severe pollution often misclassified as moderate


# 📊 Evaluation & Analysis Performed

✔ Horizon-wise MAE
✔ Per-site MAE
✔ Residual distribution
✔ Confusion matrix (pollution severity bins)
✔ PCA visualization of site embeddings
✔ Early stopping training regime
✔ Target scaling + inverse transform for real-unit evaluation

# 📂 Project Structure

Luftwächter/
│
├── src/
│   ├── models/
│   │   ├── st_transformer.py
│   │   └── graph_st_transformer.py
│   │
│   ├── data/
│   │   └── dataset.py
│   │
│   ├── utils/
│   │   └── metrics.py
│   │
│   ├── train_baseline.py
│   ├── train_graph.py
│   ├── eval_baseline_realunits.py
│   ├── eval_graph_realunits.py
│   └── plot_predictions.py
│
├── notebooks/
│   ├── 06_global_normalization.ipynb
│   └── final_analysis.ipynb
│
├── scripts/
│   ├── 01_unpack_and_inspect.py
│   ├── 02_check_all_sites.py
│   ├── 03_align_multisite.py
│   ├── 04_make_windows.py
│   ├── 05_split_windows.py
│   └── 06_save_raw_site_coords.py
│
├── configs/
│   ├── adjacency_final.npy
│   ├── site_coords_raw.npy
│   ├── final_feature_scaler.json
│   └── target_scaler.json
│
├── outputs/
│   ├── checkpoints/
│   └── figures/
│
├── data/
│   ├── processed/
│   └── raw/
│
├── requirements.txt
└── README.md

# ⚙️ Installation

1️⃣ Create environment
python -m venv .venv
source .venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt
🍎 Apple Silicon (M1/M2)

Install PyTorch separately with MPS support:
pip install torch torchvision torchaudio

Device used:
Device: mps

# 🚀 Training
Baseline ST Transformer
python -m src.train_baseline
Graph-ST (Wind-aware)
python -m src.train_graph

# 📊 Evaluation (Real Units)
python -m src.eval_graph_realunits

# 📈 Notebook Analysis

Open:
notebooks/final_analysis.ipynb

Includes:
Horizon-wise MAE plots
Per-site MAE plots
Residual histograms
Confusion matrix (O₃ severity)
PCA site embeddings

# 🧠 Technical Specifications
Parameter	                Value
Input Window (Tin)	        24 hours
Forecast Horizon (Tout)	    6 hours
Sites	                    7
Targets	                    2 (O₃, NO₂)
d_model	                    128
Attention Heads	            4
Temporal Layers	            2
Spatial Layers	            2
Dropout	                    0.1
Loss	                    SmoothL1Loss
Early Stopping	            Yes

# 🧪 Research Contribution

This project demonstrates:
Transformer-based multi-site forecasting
Graph-enhanced spatial propagation
Dynamic wind-modulated adjacency
Latent spatial representation learning
Thorough post-training error analysis
Even though graph models did not outperform baseline in MAE, they learned interpretable spatial structure — suggesting potential for further enhancement.

# 📌 Future Improvements

Stronger wind-aware adjacency weighting
Separate loss weighting for extreme pollution
Probabilistic forecasting
Uncertainty estimation
Temporal cross-attention between sites
Adaptive graph learning

# 📜 License
For academic and research use.