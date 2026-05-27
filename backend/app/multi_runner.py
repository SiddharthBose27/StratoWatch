# backend/app/multi_runner.py
from __future__ import annotations
import base64
import json
import os
import re
import subprocess
import sys
from glob import glob
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_DIR = os.path.join(ROOT, "stratowatch_multi_site")

NPZ_PATH = os.path.join(
    PROJECT_DIR, "data", "processed", "splits_final_Yscaled_Tin24_Tout6_stride1.npz"
)
SCALER_PATH = os.path.join(PROJECT_DIR, "configs", "target_scaler.json")

FIG_DIR = os.path.join(PROJECT_DIR, "outputs", "figures")
REPORT_DIR = os.path.join(PROJECT_DIR, "outputs", "figures", "ui_report")

MODEL_TO_CMD = {
    # eval scripts use the fixed NPZ + checkpoint (trained for 7 sites)
    "st_transformer": ["-m", "src.eval_baseline_realunits"],
    "graph_st_static": ["-m", "src.eval_graph_realunits"],
    "graph_st_dynamic_wind": ["-m", "src.eval_graph_realunits"],
}


def _run_py(args: List[str]) -> Tuple[int, str, str]:
    cmd = [sys.executable] + args
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _extract_metrics(text: str) -> Dict[str, float]:
    """
    Parse stdout like:
      MAE : 18.69
      RMSE: 26.66
    """
    metrics: Dict[str, float] = {}

    def grab(k: str):
        m = re.search(rf"{k}\s*[:=]\s*([-+]?\d*\.?\d+)", text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    for k in ["MAE", "RMSE", "MSE", "R2", "R²"]:
        v = grab(k)
        if v is not None:
            metrics["R2" if k == "R²" else k] = v
    return metrics


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _collect_plots() -> List[Dict[str, str]]:
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = sorted(glob(os.path.join(FIG_DIR, "*.png")))
    return [{"name": os.path.basename(p), "b64": _b64(p)} for p in paths]


def _collect_report_pngs() -> List[Dict[str, str]]:
    os.makedirs(REPORT_DIR, exist_ok=True)
    paths = sorted(glob(os.path.join(REPORT_DIR, "*.png")))
    return [{"name": os.path.basename(p), "b64": _b64(p)} for p in paths]


def _clear_old_figures():
    os.makedirs(FIG_DIR, exist_ok=True)
    for p in glob(os.path.join(FIG_DIR, "*.png")):
        try:
            os.remove(p)
        except Exception:
            pass
    os.makedirs(REPORT_DIR, exist_ok=True)
    for p in glob(os.path.join(REPORT_DIR, "*.png")):
        try:
            os.remove(p)
        except Exception:
            pass


def _inverse_scale(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # y: (B, Tout, S, C)
    return y * std.reshape(1, 1, 1, -1) + mean.reshape(1, 1, 1, -1)


def _baseline_fallback(site_count: int) -> Dict[str, float]:
    """
    Works for ANY site_count.
    Uses TEST split from NPZ and a persistence baseline:
      y_hat[:, t] = y[:, t-1]  (within horizon steps)
    Then computes MAE/RMSE in REAL units.
    """
    if not os.path.exists(NPZ_PATH):
        return {"MAE": -1.0, "RMSE": -1.0}

    d = np.load(NPZ_PATH, allow_pickle=True)
    Y = d["Y_test"]  # (B, Tout, S, C)
    Y_mask = d["Y_mask_test"]  # same shape

    B, Tout, S, C = Y.shape
    S_use = min(site_count, S)
    Y = Y[:, :, :S_use, :]
    Y_mask = Y_mask[:, :, :S_use, :].astype(np.float32)

    # persistence baseline across horizon
    Y_hat = Y.copy()
    if Tout > 1:
        Y_hat[:, 1:, :, :] = Y[:, :-1, :, :]

    # inverse scale to REAL units
    with open(SCALER_PATH, "r") as f:
        scaler = json.load(f)
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)

    Y_real = _inverse_scale(Y, mean, std)
    Y_hat_real = _inverse_scale(Y_hat, mean, std)

    diff = (Y_hat_real - Y_real) * Y_mask
    mae = np.sum(np.abs(diff)) / (np.sum(Y_mask) + 1e-8)
    rmse = np.sqrt(np.sum(diff**2) / (np.sum(Y_mask) + 1e-8))

    return {"MAE": float(mae), "RMSE": float(rmse)}


def _run_report_plots(model_name: str, site_count: int, timeout_sec: int = 30) -> Tuple[bool, str]:
    """
    Runs report plot generation in a subprocess so we can timeout safely.
    Returns (ok, message).
    """
    code = (
        "from src.report_plots import generate_ui_plots; "
        "generate_ui_plots(project_root=%r, model_name=%r, site_count=%r, max_batches=6)"
        % (PROJECT_DIR, model_name, site_count)
    )
    cmd = [sys.executable, "-c", code]
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return False, f"Plot generation timed out after {timeout_sec}s"

    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        return False, msg[:2000] if msg else "Plot generation failed"

    return True, ""


def run_multisite(site_count: int, model_name: str):
    """
    Returns:
      {ok, mode, warning?, metrics, plots, logs?}
    """
    if not os.path.isdir(PROJECT_DIR):
        return {"error": f"Project dir not found: {PROJECT_DIR}"}

    _clear_old_figures()

    # fallback when site_count != 7
    if site_count != 7:
        metrics = _baseline_fallback(site_count)
        plots = _collect_plots()
        result = {
            "ok": True,
            "mode": "baseline_fallback",
            "warning": (
                f"⚠ Trained checkpoints are for 7 sites. You selected {site_count}. "
                f"Running baseline fallback."
            ),
            "model": model_name,
            "metrics": metrics,
            "plots": plots,
        }
    else:
        if model_name not in MODEL_TO_CMD:
            return {"error": f"Unknown model_name: {model_name}"}

        rc, out, err = _run_py(MODEL_TO_CMD[model_name])
        if rc != 0:
            return {
                "error": "Model run failed",
                "stderr": err[-4000:],
                "stdout": out[-4000:],
            }

        metrics = _extract_metrics(out)
        plots = _collect_plots()

        result = {
            "ok": True,
            "mode": "checkpoint",
            "model": model_name,
            "metrics": metrics,
            "plots": plots,
            "logs": {"stdout_tail": out[-2000:], "stderr_tail": err[-2000:]},
        }

    # Optional plot pack (UI report) with timeout so UI doesn't hang
    ok, msg = _run_report_plots(model_name=model_name, site_count=site_count, timeout_sec=30)
    if ok:
        report_plots = _collect_report_pngs()
        if report_plots:
            result["plots"] = report_plots
    else:
        if msg:
            if result.get("warning"):
                result["warning"] = result["warning"] + "\n" + msg
            else:
                result["warning"] = msg

    return result
