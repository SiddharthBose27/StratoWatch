# baselines/train_rf.py
# Purpose:
#   Train RandomForestRegressor baselines on flattened window features
#   Predicts 2 outputs: [O3, NO2] (same y_seq as sequences.npz)
#   Saves:
#     - outputs/baselines/rf_predictions.npy
#     - outputs/baselines/rf_metrics.txt

import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from baselines.common_data import load_splits


def main():
    os.makedirs("outputs/baselines", exist_ok=True)

    splits = load_splits()
    X_train = splits["X_train_flat"]
    y_train = splits["y_train"]
    X_test = splits["X_test_flat"]
    y_test = splits["y_test"]

    print("✅ Loaded splits:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    # Random Forest (multi-output wrapper)
    rf = RandomForestRegressor(
    n_estimators=100,        # Reduced trees (faster)
    max_depth=15,            # Prevent deep trees (stability + speed)
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",     # Very important for high-dimensional input
    random_state=42,
    n_jobs=-1,
)
    model = MultiOutputRegressor(rf)

    print("🚀 Training Random Forest...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print("\n✅ RF TEST METRICS")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    # Per-target
    for i, name in enumerate(["O3", "NO2"]):
        mae_i = mean_absolute_error(y_test[:, i], preds[:, i])
        rmse_i = np.sqrt(mean_squared_error(y_test[:, i], preds[:, i]))
        r2_i = r2_score(y_test[:, i], preds[:, i])
        print(f"\n📌 RF Metrics for {name}")
        print(f"MAE : {mae_i:.4f}")
        print(f"RMSE: {rmse_i:.4f}")
        print(f"R2  : {r2_i:.4f}")

    # Save predictions + metrics
    np.save("outputs/baselines/rf_preds.npy", preds)
    np.save("outputs/baselines/rf_true.npy", y_test)

    with open("outputs/baselines/rf_metrics.txt", "w") as f:
        f.write("RF TEST METRICS\n")
        f.write(f"MAE={mae}\nMSE={mse}\nRMSE={rmse}\nR2={r2}\n")

    print("\n✅ Saved RF outputs to outputs/baselines/")


if __name__ == "__main__":
    main()