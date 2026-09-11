from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .multi_runner import run_multisite
from .runner import run_single_site_real


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="StratoWatch API",
    version="2.0",
)


# ============================================================
# CORS
# ============================================================

_CORS_ENV = os.environ.get("CORS_ALLOWED_ORIGINS", "")
if _CORS_ENV.strip():
    _CORS_ORIGINS: list[str] | str = [
        o.strip() for o in _CORS_ENV.split(",") if o.strip()
    ]
else:
    # Allow all origins by default so the Vercel-deployed frontend
    # can reach a separately-deployed backend without extra config.
    # Tighten this in production by setting CORS_ALLOWED_ORIGINS.
    _CORS_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

APP_DIR = Path(
    __file__
).resolve().parent

BACKEND_DIR = APP_DIR.parent

UPLOADS_DIR = (
    BACKEND_DIR / "uploads"
)

UPLOADS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "stratowatch-api",
        "version": "2.0",
    }


# ============================================================
# SINGLE-SITE
# ============================================================

@app.post("/api/single-site/run")
async def single_site_run(
    file: UploadFile = File(...),
    model_name: str = Form(...),
):
    """
    Run/display the frozen final single-site evaluation.

    The uploaded CSV is stored for interface compatibility.
    The official Phase 7 evaluation artifact remains the
    research evaluation source.
    """

    # --------------------------------------------------------
    # Basic filename validation
    # --------------------------------------------------------

    original_name = (
        Path(file.filename or "upload.csv").name
    )

    if not original_name.lower().endswith(
        ".csv"
    ):
        return {
            "ok": False,
            "error": (
                "Only CSV files are accepted."
            ),
        }

    # --------------------------------------------------------
    # Generate collision-safe upload path
    # --------------------------------------------------------

    upload_name = (
        f"{uuid.uuid4().hex}_{original_name}"
    )

    save_path = (
        UPLOADS_DIR / upload_name
    )

    try:
        contents = await file.read()

        with save_path.open(
            "wb"
        ) as output:
            output.write(contents)

    except Exception as exc:
        return {
            "ok": False,
            "error": (
                f"Unable to save uploaded file: {exc}"
            ),
        }

    # --------------------------------------------------------
    # Run frozen evaluation artifact
    # --------------------------------------------------------

    result = run_single_site_real(
        uploaded_csv_path=str(
            save_path
        ),
        model_name=model_name,
        include_confusion=False,
    )

    if "error" in result:
        return {
            "ok": False,
            **result,
        }

    return {
        "ok": True,
        **result,
        "uploaded_filename": original_name,
    }


# ============================================================
# MULTI-SITE
# ============================================================

@app.post("/api/multi-site/run")
async def multi_site_run(
    site_count: int = Form(...),
    model_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Run the final multi-site evaluation.

    Uploaded files are currently accepted for API/interface
    compatibility. The frozen seven-site research evaluation
    continues to use the validated research artifact.
    """

    # --------------------------------------------------------
    # Validate site count
    # --------------------------------------------------------

    if site_count < 2:
        return {
            "ok": False,
            "error": (
                "site_count must be at least 2."
            ),
        }

    if len(files) != site_count:
        return {
            "ok": False,
            "error": (
                f"Expected {site_count} uploaded files, "
                f"received {len(files)}."
            ),
        }

    # --------------------------------------------------------
    # Validate uploaded files
    # --------------------------------------------------------

    invalid_files = []

    for uploaded in files:
        filename = Path(
            uploaded.filename or ""
        ).name

        if not filename.lower().endswith(
            ".csv"
        ):
            invalid_files.append(
                filename or "<unnamed file>"
            )

    if invalid_files:
        return {
            "ok": False,
            "error": (
                "All uploaded site files must be CSV files."
            ),
            "invalid_files": invalid_files,
        }

    # --------------------------------------------------------
    # Save this request's inputs.  Non-seven-site fallback plotting consumes
    # these paths directly, so it cannot accidentally fall back to the
    # frozen research arrays.
    # --------------------------------------------------------
    uploaded_paths: List[str] = []
    uploaded_site_names: List[str] = []

    try:
        for index, uploaded in enumerate(files, start=1):
            original_name = Path(uploaded.filename or f"site_{index}.csv").name
            saved_path = UPLOADS_DIR / f"{uuid.uuid4().hex}_{index}_{original_name}"
            with saved_path.open("wb") as output:
                output.write(await uploaded.read())
            uploaded_paths.append(str(saved_path))
            uploaded_site_names.append(Path(original_name).stem)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Unable to save uploaded site data: {exc}",
        }

    # --------------------------------------------------------
    # Run multi-site evaluation / request-local fallback
    # --------------------------------------------------------

    result = run_multisite(
        site_count=site_count,
        model_name=model_name,
        uploaded_csv_paths=uploaded_paths,
        uploaded_site_names=uploaded_site_names,
    )

    if "error" in result:
        return {
            "ok": False,
            **result,
        }

    # multi_runner already returns ok=True on success.
    return {
        "ok": True,
        **result,
        "uploaded_file_count": len(files),
    }


# ============================================================
# OPTIONAL STATIC ARTIFACT MOUNT
# ============================================================

# This exposes generated research/evaluation artifacts if
# future frontend logic needs direct URLs.
#
# Existing frontend base64 plot handling does not depend on
# this mount, so this is intentionally non-invasive.

SINGLE_SITE_OUTPUTS = (
    BACKEND_DIR
    / "stratowatch_single_site"
    / "outputs"
)

if SINGLE_SITE_OUTPUTS.exists():
    app.mount(
        "/artifacts/single-site",
        StaticFiles(
            directory=str(
                SINGLE_SITE_OUTPUTS
            )
        ),
        name="single-site-artifacts",
    )
