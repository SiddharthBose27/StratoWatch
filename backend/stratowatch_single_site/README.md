# 🌍 Luftwächter (Single Site)
Deep Residual Transformer for Air Quality Prediction

Project Type: Spatio-Temporal Air Pollution Forecasting
Author: Siddharth Bose


# 🧠 Project Overview

Luftwächter (German: “Air Guardian”) is a deep learning–based air quality prediction framework designed to improve atmospheric pollutant forecasts at a single monitoring site.

Instead of directly predicting pollutant concentrations, the system learns the residual error between official forecast values and real ground truth measurements:

y=forecast+Δ

Where:
forecast = physics-based meteorological forecast
Δ = learned correction from ML model
This residual-learning formulation significantly improves predictive accuracy.

# 🎯 Target Variables

O3_target (Ozone concentration)
NO2_target (Nitrogen Dioxide concentration)

# 🏗 Model Architecture

Primary Model:
Temporal Transformer Encoder
Multi-head self-attention
Residual learning framework

Baseline Models:
1.Random Forest
2.XGBoost
3.LSTM
4.Temporal Convolutional Network (TCN)
5.Forecast-only baseline


⚙️ Data Pipeline

1️⃣ Data Cleaning
Time reindexing (hourly grid)
Missing value interpolation
Forward/backward fill
Outlier clipping (1%–99% quantiles)

2️⃣ Feature Engineering
Cyclical encoding (hour, month, day-of-week)
Wind magnitude & direction features
Lag features (1,2,3,6,12,24)
Rolling statistics (mean/std)
Satellite interaction signals

3️⃣ Scaling
Feature scaling via grouped StandardScaler
Residual target scaling (saved as y_res_scaler.pkl)

4️⃣ Sliding Window Sequences
Input shape:
(N, 24, F)
Where:
24 = past 24 hours
F = engineered features

📊 Experimental Setup

Train/Val/Test Split: 70% / 15% / 15%
Loss Function: MSE
Optimizer: AdamW
Early Stopping: patience=5
Hardware: Apple M1 (MPS)

📈 Final Results (Real Units) 

Model	                     MAE	                    RMSE	                 R²
Forecast Baseline	         30.78	                    42.17	                -0.27
Random Forest	               —	                     —	                     0.316
LSTM	                     21.60	                    31.50	                 0.184
TCN	                         21.22	                    29.93	                 0.293
Transformer	                ~20–21	                   ~30	                    ~0.26–0.33
XGBoost	                     17.95	                    26.69	                 0.4208

# 🏆 Best Model: XGBoost (Residual Learning)

# 📉 Additional Evaluation

1️⃣ Loss Curves
Transformer loss curve
LSTM loss curve
TCN loss curve

Saved in:
outputs/plots/

2️⃣ Confusion Matrices
Continuous predictions converted to air-quality categories:
Low
Medium
High

Saved in:
outputs/confusion_matrices/

# 🧪 Scientific Findings
1.Residual learning drastically improves forecast baseline.
2.Gradient boosting (XGBoost) outperforms deep sequential models.
3.Pure LSTM underperforms attention and boosting.
4.Transformer competitive but not dominant.
5.Feature engineering plays a major role.

# 🗂 Project Structure

Luftwächter(Single site)/
├── data/
│   ├── sequences.npz
│   ├── y_res_scaler.pkl
│   ├── feature_list.json
│
├── models/
│   └── temporal_transformer.py
│
├── baselines/
│   ├── train_rf.py
│   ├── train_xgb.py
│   ├── train_lstm.py
│   ├── train_tcn.py
│   └── confusion_matrices.py
│
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   ├── baselines/
│   ├── confusion_matrices/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md


# 🚀 How to Run

Install Dependencies
pip install -r requirements.txt

Train Transformer
python train.py

Evaluate Transformer
python evaluate.py

Train Baselines
python -m baselines.train_rf
python -m baselines.train_xgb
python -m baselines.train_lstm
python -m baselines.train_tcn

Generate Confusion Matrices
python -m baselines.confusion_matrices

# 📚 Research Contribution

This project demonstrates:
Residual correction of physics-based forecasts
Comprehensive comparison of deep learning vs boosting
Real-unit evaluation (MAE, RMSE, R²)
Classification reinterpretation via confusion matrices

# 🔮 Future Work (Tier-1 Extension)

Multi-site spatial graph modeling
Graph Neural Networks (GNN)
Multi-site domain adaptation
Transfer learning across stations
Attention heatmap interpretability

# 🏁 Project Status

✅ Single-site pipeline complete
✅ Multi-model comparison complete
✅ Paper-ready evaluation complete
Luftwächter (Single Site) is experimentally complete.