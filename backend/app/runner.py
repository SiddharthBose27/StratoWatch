import base64
import os
import re
import subprocess
import sys
from glob import glob
from typing import Dict, List, Tuple

MAX_PLOTS = 6  # start small, increase later

def _collect_pngs() -> List[Dict[str, str]]:
    paths = []
    paths += sorted(glob(os.path.join(PLOTS_DIR, "*.png")))
    paths += sorted(glob(os.path.join(CM_DIR, "*.png")))

    # take latest plots only (most recent files)
    paths = sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)[:MAX_PLOTS]

    plots = []
    for p in paths:
        plots.append({"name": os.path.basename(p), "b64": _b64(p)})
    return plots

# Path: backend/stratowatch_single_site
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_DIR = os.path.join(ROOT, "stratowatch_single_site")

# Where your scripts write images (from your README)
PLOTS_DIR = os.path.join(PROJECT_DIR, "outputs", "plots")
CM_DIR = os.path.join(PROJECT_DIR, "outputs", "confusion_matrices")

# Your pipeline expects this exact file name (you confirmed overwrite)
UNSEEN_FILE = os.path.join(PROJECT_DIR, "site_1_unseen_input_data.csv")


MODEL_TO_CMD = {
    # You can rename keys to match your frontend dropdown values
    "transformer": ["evaluate.py"],
    "xgboost_residual": ["-m", "baselines.train_xgb"],
    "lstm": ["-m", "baselines.train_lstm"],
    "tcn": ["-m", "baselines.train_tcn"],
    # optional: "forecast_baseline": ["-m", "baselines.forecast_baseline"],
}


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _run_py(args: List[str]) -> Tuple[int, str, str]:
    """
    Runs python inside PROJECT_DIR so relative paths work.
    Uses current venv python.
    """
    cmd = [sys.executable] + args
    env = os.environ.copy()

    # Ensure PROJECT_DIR is importable for "-m baselines.*"
    env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")

    p = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout, p.stderr


def _extract_metrics(text: str) -> Dict[str, float]:
    """
    Tries to parse printed metrics from stdout.
    Works with patterns like:
      MAE : 17.95
      RMSE: 26.69
      R2  : 0.4208
    """
    metrics = {}

    def find_one(key: str) -> float | None:
        # match "KEY : value" or "KEY: value"
        m = re.search(rf"{key}\s*[:=]\s*([-+]?\d*\.?\d+)", text, re.IGNORECASE)
        if not m:
            return None
        return float(m.group(1))

    for k in ["MAE", "MSE", "RMSE", "R2", "R²"]:
        v = find_one(k)
        if v is not None:
            if k == "R²":
                metrics["R2"] = v
            else:
                metrics[k] = v

    # Normalize
    if "R²" in metrics:
        metrics["R2"] = metrics.pop("R²")

    return metrics


def _collect_pngs() -> List[Dict[str, str]]:
    paths = []
    paths += sorted(glob(os.path.join(PLOTS_DIR, "*.png")))
    paths += sorted(glob(os.path.join(CM_DIR, "*.png")))

    plots = []
    for p in paths:
        plots.append({"name": os.path.basename(p), "b64": _b64(p)})
    return plots


def run_single_site_real(uploaded_csv_path: str, model_name: str, include_confusion: bool = True):
    if not os.path.isdir(PROJECT_DIR):
        return {"error": f"Project dir not found: {PROJECT_DIR}"}

    if model_name not in MODEL_TO_CMD:
        return {"error": f"Unknown model_name: {model_name}"}

    # 1) Overwrite the file your code expects
    os.makedirs(os.path.dirname(UNSEEN_FILE), exist_ok=True)
    with open(uploaded_csv_path, "rb") as src, open(UNSEEN_FILE, "wb") as dst:
        dst.write(src.read())

    # 2) Run selected model
    args = MODEL_TO_CMD[model_name]
    rc, out, err = _run_py(args)

    if rc != 0:
        return {
            "error": "Model run failed",
            "model": model_name,
            "stderr": err[-4000:],
            "stdout": out[-4000:],
        }

    # 3) Optionally run confusion matrices
    cm_out = ""
    cm_err = ""
    if include_confusion:
        rc2, out2, err2 = _run_py(["-m", "baselines.confusion_matrices"])
        cm_out, cm_err = out2, err2
        # If confusion fails, we still return main model results
        # so we do NOT hard-fail the API.

    # 4) Parse metrics from output
    metrics = _extract_metrics(out)

    # 5) Collect plots as base64
    plots = _collect_pngs()

    return {
        "model": model_name,
        "metrics": metrics,
        "plots": plots,
        "logs": {
            "stdout_tail": out[-2500:],
            "stderr_tail": err[-2500:],
            "cm_stdout_tail": cm_out[-1500:],
            "cm_stderr_tail": cm_err[-1500:],
        },
    }