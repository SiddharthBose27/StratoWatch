# backend/stratowatch_multi_site/src/report_plots.py
from __future__ import annotations

import os
import json
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _safe_load_npz(npz_path: str) -> Dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _inverse_scale(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # y: (B, Tout, S, C)
    return y * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def _masked_mae(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: (B, Tout, S, C)
    denom = mask.sum().clamp(min=1.0)
    return (torch.abs(pred - true) * mask).sum() / denom


def _masked_rmse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return torch.sqrt(((pred - true) ** 2 * mask).sum() / denom)


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """
    Tiny PCA -> 2D using SVD (no sklearn).
    X: (S, D)
    returns (S, 2)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    return Z


def _severity_bins_o3(x: np.ndarray) -> np.ndarray:
    """
    Simple severity bins for O3. You can tune thresholds.
    0: low, 1: moderate, 2: high, 3: very high
    """
    # thresholds in "real units" (whatever your target units are after inverse scaling)
    bins = np.array([0.0, 60.0, 120.0, 180.0, 1e9], dtype=float)
    return np.digitize(x, bins) - 1  # 0..3


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < k and 0 <= p < k:
            cm[t, p] += 1
    return cm


def generate_ui_plots(
    project_root: str,
    model_name: str,
    site_count: int,
    max_batches: int = 6,
) -> List[str]:
    """
    Generates a "final_analysis-like" pack of plots as PNGs.
    Returns list of saved PNG paths.

    NOTE:
    - Real plots are only generated when site_count == 7 (trained config).
    - If fallback (site_count != 7) we still generate a small info plot.
    """

    out_dir = os.path.join(project_root, "outputs", "figures", "ui_report")
    _ensure_dir(out_dir)

    # -------- Fallback plot if not 7 sites --------
    if site_count != 7:
        p = os.path.join(out_dir, "00_notice_fallback.png")
        plt.figure(figsize=(10, 3))
        plt.axis("off")
        plt.text(
            0.02,
            0.55,
            f"Fallback Mode\n\nTrained checkpoints are for 7 sites.\nYou selected {site_count} sites.",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        return [p]

    # -------- Load dataset + scaler --------
    npz_path = os.path.join(project_root, "data", "processed", "splits_final_Yscaled_Tin24_Tout6_stride1.npz")
    scaler_path = os.path.join(project_root, "configs", "target_scaler.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    d = _safe_load_npz(npz_path)
    X_test = d["X_test"]            # (N, Tin, S, F)
    Y_test = d["Y_test"]            # (N, Tout, S, C)
    Y_mask_test = d["Y_mask_test"]  # (N, Tout, S, C)
    def _as_int(x):
        try:
            # numpy scalar or 0-d array
            return int(x.item()) if hasattr(x, "item") else int(x)
        except Exception:
            return int(x)

    tin = _as_int(d["tin"])
    tout = _as_int(d["tout"])
    sites = d["sites"]

    with open(scaler_path, "r") as f:
        sc = json.load(f)
    mean = torch.tensor(sc["mean"], dtype=torch.float32)
    std = torch.tensor(sc["std"], dtype=torch.float32)

    # -------- Load model --------
    device = _device()
    mean = mean.to(device)
    std = std.to(device)

    # Import models lazily (so runner import is light)
    from src.models.st_transformer import STTransformer
    from src.models.graph_st_transformer import GraphSTTransformer

    num_features = X_test.shape[-1]
    num_sites = X_test.shape[2]

    # checkpoints
    ckpt_dir = os.path.join(project_root, "outputs", "checkpoints")
    st_ckpt = os.path.join(ckpt_dir, "st_transformer_best.pt")
    graph_ckpt = os.path.join(ckpt_dir, "graph_st_best.pt")

    # dynamic wind checkpoint naming can differ — try a few common names
    dyn_candidates = [
        os.path.join(ckpt_dir, "dynamic_wind_graph_st_best.pt"),
        os.path.join(ckpt_dir, "graph_st_dynamic_best.pt"),
        os.path.join(ckpt_dir, "graph_st_dynamic_wind_best.pt"),
    ]
    dyn_ckpt = next((p for p in dyn_candidates if os.path.exists(p)), None)

    # adjacency + coords for graph models
    coords_path = os.path.join(project_root, "configs", "site_coords_raw.npy")
    A_path = os.path.join(project_root, "data", "processed", "adjacency_final.npy")
    if not os.path.exists(A_path):
        # also try configs
        A_path2 = os.path.join(project_root, "configs", "adjacency_final.npy")
        if os.path.exists(A_path2):
            A_path = A_path2

    site_coords = None
    A = None
    if os.path.exists(coords_path):
        site_coords = torch.tensor(np.load(coords_path), dtype=torch.float32, device=device)
    if os.path.exists(A_path):
        A = torch.tensor(np.load(A_path), dtype=torch.float32, device=device)

    # Build model and load weights
    if model_name == "st_transformer":
        if not os.path.exists(st_ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {st_ckpt}")

        model = STTransformer(
            num_features=num_features,
            tin=tin,
            tout=tout,
            num_sites=num_sites,
        ).to(device)

        ckpt = torch.load(st_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        def forward_fn(Xb: torch.Tensor) -> torch.Tensor:
            Y_hat, _ = model(Xb)
            return Y_hat

    elif model_name in ("graph_st_static", "graph_st_dynamic_wind"):
        chosen = graph_ckpt if model_name == "graph_st_static" else dyn_ckpt or graph_ckpt
        if not chosen or not os.path.exists(chosen):
            raise FileNotFoundError(f"Graph checkpoint not found for {model_name}")

        if A is None or site_coords is None:
            raise FileNotFoundError("Graph model needs adjacency + site_coords files.")

        model = GraphSTTransformer(
            num_features=num_features,
            tin=tin,
            tout=tout,
            num_sites=num_sites,
        ).to(device)

        ckpt = torch.load(chosen, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        def forward_fn(Xb: torch.Tensor) -> torch.Tensor:
            # NOTE: If your dynamic-wind model needs different args internally,
            # keep it same for now; weights will reflect dynamic training.
            return model(Xb, A, site_coords)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # -------- Run a few batches for plots (fast) --------
    # Use only a subset to keep API fast
    N = X_test.shape[0]
    bs = 64
    batches = min(max_batches, math.ceil(N / bs))

    maes_h = np.zeros((tout,), dtype=float)
    rmses_h = np.zeros((tout,), dtype=float)
    counts_h = np.zeros((tout,), dtype=float)

    # per-site MAE (avg across horizons + targets)
    mae_site = np.zeros((num_sites,), dtype=float)
    cnt_site = np.zeros((num_sites,), dtype=float)

    # residuals (sample)
    residuals = []

    # confusion (O3 only, horizon 0)
    ytrue_bins = []
    ypred_bins = []

    # store a small prediction sample for a plot
    sample_pred = None
    sample_true = None

    with torch.no_grad():
      for bi in range(batches):
        i0 = bi * bs
        i1 = min(N, (bi + 1) * bs)

        Xb = torch.tensor(X_test[i0:i1], dtype=torch.float32, device=device)
        Yb = torch.tensor(Y_test[i0:i1], dtype=torch.float32, device=device)
        Mb = torch.tensor(Y_mask_test[i0:i1], dtype=torch.float32, device=device)

        Yhat = forward_fn(Xb)

        # inverse scale to real units
        Yhat_r = _inverse_scale(Yhat, mean, std)
        Y_r = _inverse_scale(Yb, mean, std)

        # horizon-wise metrics
        for h in range(tout):
            mh = Mb[:, h:h+1, :, :]  # (B,1,S,C)
            predh = Yhat_r[:, h:h+1, :, :]
            trueh = Y_r[:, h:h+1, :, :]
            maes_h[h] += float(_masked_mae(predh, trueh, mh).item())
            rmses_h[h] += float(_masked_rmse(predh, trueh, mh).item())
            counts_h[h] += 1.0

        # per-site MAE across horizons+targets
        # compute abs error masked: (B,Tout,S,C)
        abs_err = torch.abs(Yhat_r - Y_r) * Mb
        # sum over B,Tout,C -> per-site
        site_sum = abs_err.sum(dim=(0,1,3)).detach().cpu().numpy()
        site_cnt = Mb.sum(dim=(0,1,3)).clamp(min=1.0).detach().cpu().numpy()
        mae_site += site_sum / site_cnt
        cnt_site += 1.0

        # residuals sample (take O3 only)
        # residuals for all points where mask present
        res = (Yhat_r - Y_r).detach().cpu().numpy()
        m = Mb.detach().cpu().numpy().astype(bool)
        # O3 channel 0
        r_o3 = res[..., 0][m[..., 0]]
        if r_o3.size > 0:
            residuals.append(r_o3[:20000])

        # confusion: horizon 0, O3 only, average across sites
        y0_true = Y_r[:, 0, :, 0].detach().cpu().numpy()  # (B,S)
        y0_pred = Yhat_r[:, 0, :, 0].detach().cpu().numpy()
        m0 = Mb[:, 0, :, 0].detach().cpu().numpy().astype(bool)
        # flatten masked
        yt = y0_true[m0]
        yp = y0_pred[m0]
        if yt.size > 0:
            ytrue_bins.append(_severity_bins_o3(yt))
            ypred_bins.append(_severity_bins_o3(yp))

        # store sample for pred plot
        if sample_pred is None:
            sample_pred = Yhat_r[0].detach().cpu().numpy()  # (Tout,S,C)
            sample_true = Y_r[0].detach().cpu().numpy()

    maes_h = maes_h / np.clip(counts_h, 1.0, None)
    rmses_h = rmses_h / np.clip(counts_h, 1.0, None)
    mae_site = mae_site / np.clip(cnt_site, 1.0, None)

    residuals = np.concatenate(residuals) if len(residuals) else np.array([], dtype=float)
    ytrue_bins = np.concatenate(ytrue_bins) if len(ytrue_bins) else np.array([], dtype=int)
    ypred_bins = np.concatenate(ypred_bins) if len(ypred_bins) else np.array([], dtype=int)

    saved: List[str] = []

    # -------- Plot 1: Horizon-wise MAE/RMSE --------
    p1 = os.path.join(out_dir, "01_horizon_mae_rmse.png")
    plt.figure(figsize=(10, 4))
    x = np.arange(1, tout+1)
    plt.plot(x, maes_h, marker="o", label="MAE")
    plt.plot(x, rmses_h, marker="o", label="RMSE")
    plt.title("Horizon-wise Error (Real Units)")
    plt.xlabel("Forecast Horizon (step)")
    plt.ylabel("Error")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()
    saved.append(p1)

    # -------- Plot 2: Per-site MAE --------
    p2 = os.path.join(out_dir, "02_per_site_mae.png")
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(num_sites), mae_site)
    plt.title("Per-site MAE (avg across horizons + targets)")
    plt.xlabel("Site index")
    plt.ylabel("MAE")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()
    saved.append(p2)

    # -------- Plot 3: Residual histogram (O3) --------
    p3 = os.path.join(out_dir, "03_residual_hist_o3.png")
    plt.figure(figsize=(10, 4))
    if residuals.size:
        plt.hist(residuals, bins=80)
        plt.title("Residual Distribution (O3) — Pred - True")
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.25)
    else:
        plt.axis("off")
        plt.text(0.1, 0.5, "No residuals collected.", fontsize=14)
    plt.tight_layout()
    plt.savefig(p3, dpi=160)
    plt.close()
    saved.append(p3)

    # -------- Plot 4: Confusion matrix (O3 severity bins) --------
    p4 = os.path.join(out_dir, "04_confusion_o3_bins.png")
    plt.figure(figsize=(6, 5))
    if ytrue_bins.size and ypred_bins.size:
        cm = _confusion_matrix(ytrue_bins, ypred_bins, k=4)
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (O3 Severity Bins)")
        plt.xlabel("Predicted bin")
        plt.ylabel("True bin")
        for i in range(4):
            for j in range(4):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xticks([0,1,2,3], ["Low","Mod","High","V.High"])
        plt.yticks([0,1,2,3], ["Low","Mod","High","V.High"])
    else:
        plt.axis("off")
        plt.text(0.1, 0.5, "No confusion data collected.", fontsize=14)
    plt.tight_layout()
    plt.savefig(p4, dpi=160)
    plt.close()
    saved.append(p4)

    # -------- Plot 5: “Embedding” PCA proxy (per-site profile PCA) --------
    # If your notebook used learned embeddings, we approximate with per-site MAE + horizon profile.
    p5 = os.path.join(out_dir, "05_site_pca_proxy.png")
    plt.figure(figsize=(7, 6))
    # build a simple site feature: [mae_site, mean_target(approx)] - we can only use mae_site reliably here
    site_feat = np.stack([mae_site, np.arange(num_sites)], axis=1).astype(float)
    Z = _pca_2d(site_feat)
    plt.scatter(Z[:, 0], Z[:, 1])
    for i in range(num_sites):
        plt.text(Z[i, 0], Z[i, 1], f"S{i}", fontsize=10)
    plt.title("Site Embedding (PCA proxy)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(p5, dpi=160)
    plt.close()
    saved.append(p5)

    # -------- Plot 6: Prediction vs GT sample (O3, site 0) --------
    p6 = os.path.join(out_dir, "06_pred_vs_gt_sample_o3_site0.png")
    plt.figure(figsize=(10, 4))
    if sample_pred is not None and sample_true is not None:
        # (Tout,S,C)
        pred = sample_pred[:, 0, 0]
        tru = sample_true[:, 0, 0]
        x = np.arange(1, tout + 1)
        plt.plot(x, tru, marker="o", label="GT")
        plt.plot(x, pred, marker="o", label="Pred")
        plt.title("Prediction vs Ground Truth (Sample) — O3, Site 0")
        plt.xlabel("Horizon step")
        plt.ylabel("O3 (real units)")
        plt.grid(True, alpha=0.25)
        plt.legend()
    else:
        plt.axis("off")
        plt.text(0.1, 0.5, "No sample captured.", fontsize=14)
    plt.tight_layout()
    plt.savefig(p6, dpi=160)
    plt.close()
    saved.append(p6)

    return saved