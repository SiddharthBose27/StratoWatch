# baselines/confusion_matrices.py
# Purpose:
#   Convert regression predictions (O3, NO2) into classification labels
#   and plot confusion matrices using seaborn.
#
# Run:
#   python3 -m baselines.confusion_matrices

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# -----------------------------
# 1) Define bins -> classes
# -----------------------------
# You can change these thresholds later (CPCB/WHO etc.)
# For now we use 3 classes (Low/Medium/High)
O3_BINS  = [0, 50, 100, 1e9]    # 0–50, 50–100, >100
NO2_BINS = [0, 40, 80, 1e9]     # 0–40, 40–80, >80

CLASS_NAMES = ["Low", "Medium", "High"]


def to_classes(y, bins):
    """
    y: (N,) continuous
    bins: list like [0, a, b, inf]
    returns class ids {0,1,2}
    """
    return np.digitize(y, bins) - 1


# -----------------------------
# 2) Plot helper
# -----------------------------
def plot_cm(y_true_cls, y_pred_cls, title, save_path):
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1, 2])

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("✅ Saved:", save_path)


def main():
    os.makedirs("outputs/confusion_matrices", exist_ok=True)

    # -----------------------------
    # 3) Load model prediction files
    # -----------------------------
    models = {
        "RF":  ("outputs/baselines/rf_preds.npy", "outputs/baselines/rf_true.npy"),
        "XGB": ("outputs/baselines/xgb_preds_real.npy", "outputs/baselines/xgb_true_real.npy"),
        "LSTM":("outputs/baselines/lstm_preds_real.npy","outputs/baselines/lstm_true_real.npy"),
        "TCN": ("outputs/baselines/tcn_preds_real.npy", "outputs/baselines/tcn_true_real.npy"),
    }

    # If you want Transformer confusion matrices too:
    # Put these files into outputs/baselines/ or outputs/
    # and uncomment/add:
    #
    # models["Transformer"] = ("outputs/transformer_preds_real.npy", "outputs/transformer_true_real.npy")

    for model_name, (pred_path, true_path) in models.items():
        if not (os.path.exists(pred_path) and os.path.exists(true_path)):
            print(f"⚠️ Missing files for {model_name}, skipping.")
            continue

        preds = np.load(pred_path)
        trues = np.load(true_path)

        # preds/trues shape: (N, 2) => [O3, NO2]
        o3_pred, no2_pred = preds[:, 0], preds[:, 1]
        o3_true, no2_true = trues[:, 0], trues[:, 1]

        # Convert to classes
        o3_pred_cls = to_classes(o3_pred, O3_BINS)
        o3_true_cls = to_classes(o3_true, O3_BINS)

        no2_pred_cls = to_classes(no2_pred, NO2_BINS)
        no2_true_cls = to_classes(no2_true, NO2_BINS)

        # Plot confusion matrices
        plot_cm(
            o3_true_cls, o3_pred_cls,
            title=f"{model_name} Confusion Matrix (O3)",
            save_path=f"outputs/confusion_matrices/{model_name}_O3_cm.png"
        )

        plot_cm(
            no2_true_cls, no2_pred_cls,
            title=f"{model_name} Confusion Matrix (NO2)",
            save_path=f"outputs/confusion_matrices/{model_name}_NO2_cm.png"
        )

    print("\n✅ Done. Confusion matrices saved in outputs/confusion_matrices/")


if __name__ == "__main__":
    main()