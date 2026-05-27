from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os

from .multi_runner import run_multisite
from .runner import run_single_site_real

app = FastAPI(title="StratoWatch API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,  # keep False unless you use cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/single-site/run")
async def single_site_run(
    file: UploadFile = File(...),
    model_name: str = Form(...),
):
    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    save_path = os.path.join(uploads_dir, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    result = run_single_site_real(
        uploaded_csv_path=save_path,
        model_name=model_name,
        include_confusion=True
    )

    if "error" in result:
        return {"ok": False, **result}

    return {"ok": True, **result}

@app.post("/api/multi-site/run")
async def multi_site_run(
    site_count: int = Form(...),
    model_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    # files are not used yet; pipeline uses preprocessed NPZ
    result = run_multisite(site_count=site_count, model_name=model_name)

    if "error" in result:
        return {"ok": False, **result}

    return result
