# baselines/train_xgb.py
# Purpose:
#   Train an XGBoost baseline on flattened sequences (24xF -> 24*F).
#   Predicts 2 outputs: [O3_residual_scaled, NO2_residual_scaled]
#   Then (optionally) reconstructs real targets using forecast + residual.
#
# Run:
#   python3 -m baselines.train_xgb

import os
import json
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from baselines.common_data import load_splits, load_feature_list_and_forecast_from_X


# XGBoost
from xgboost import XGBRegressor


def metrics_block(y_true, y_pred, title=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    out = []
    if title:
        out.append(title)
    out.append(f"MAE : {mae:.4f}")
    out.append(f"MSE : {mse:.4f}")
    out.append(f"RMSE: {rmse:.4f}")
    out.append(f"R2  : {r2:.4f}")
    return "\n".join(out), (mae, mse, rmse, r2)


def main():
    os.makedirs("outputs/baselines", exist_ok=True)

    # 1) Load split data (scaled residual targets)
    splits = load_splits()
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    print("✅ Loaded splits:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    # 2) Build 2 independent regressors (multioutput for XGB is messy)
    #    Optimized params for your high-dimensional data (4296 features)
    xgb_params = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.2,   # Very important for 4296 features
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

    print("\n🚀 Training XGB for O3 residual...")
    model_o3 = XGBRegressor(**xgb_params)
    model_o3.fit(X_train, y_train[:, 0])

    print("🚀 Training XGB for NO2 residual...")
    model_no2 = XGBRegressor(**xgb_params)
    model_no2.fit(X_train, y_train[:, 1])

    # 3) Predict scaled residuals
    pred_o3 = model_o3.predict(X_test).reshape(-1, 1)
    pred_no2 = model_no2.predict(X_test).reshape(-1, 1)
    preds_scaled = np.hstack([pred_o3, pred_no2])

    print("\n✅ Predicted scaled residuals:", preds_scaled.shape)

    # 4) Convert residuals back to real units
    y_res_scaler = joblib.load("data/y_res_scaler.pkl")
    preds_residual = y_res_scaler.inverse_transform(preds_scaled)
    trues_residual = y_res_scaler.inverse_transform(y_test)

    # 5) Reconstruct REAL target values: y = forecast + residual
    #    We need forecast extracted from test X sequences.
    feature_list, forecast = load_feature_list_and_forecast_from_X(
        X_test_seq_original=splits["X_test_seq"],  # (N_test, T, F)
        feature_list_path="data/feature_list.json",
        o3_feature_name="O3_forecast",
        no2_feature_name="NO2_forecast"
    )

    preds_real = forecast + preds_residual
    trues_real = forecast + trues_residual

    # 6) Metrics (real units)
    block, (mae, mse, rmse, r2) = metrics_block(trues_real, preds_real, "✅ XGB TEST METRICS (Real Units, Residual Learning)")
    print("\n" + block)

    # Per-target
    for i, name in enumerate(["O3_target", "NO2_target"]):
        blk_i, _ = metrics_block(trues_real[:, i], preds_real[:, i], f"\n📌 XGB Metrics for {name}")
        print(blk_i)

    # 7) Save outputs
    np.save("outputs/baselines/xgb_preds_real.npy", preds_real)
    np.save("outputs/baselines/xgb_true_real.npy", trues_real)

    with open("outputs/baselines/xgb_metrics.txt", "w") as f:
        f.write(block + "\n")

    # Save small CSV line for paper table
    with open("outputs/baselines/xgb_paper_row.csv", "w") as f:
        f.write("Method,MAE,RMSE,R2\n")
        f.write(f"XGBoost (Residual + Forecast),{mae:.6f},{rmse:.6f},{r2:.6f}\n")

    print("\n✅ Saved XGB outputs to outputs/baselines/")


if __name__ == "__main__":
    main()