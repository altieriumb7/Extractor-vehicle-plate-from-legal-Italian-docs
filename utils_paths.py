from __future__ import annotations

from pathlib import Path
from utils_bootstrap import app_tmp_dir, ensure_dir

# Base app tmp directory: /tmp/lpv_app
BASE_TMP = Path(app_tmp_dir())

# Temporary I/O paths (no writes inside repo)
UPLOAD_DIR = ensure_dir(Path(app_tmp_dir("uploads")))
PLATES_DIR = ensure_dir(Path(app_tmp_dir("plates")))

# Optional model cache dirs kept in /tmp for compatibility
MODELS_DIR = ensure_dir(Path(app_tmp_dir("models")))
PADDLE_MODELS_DIR = ensure_dir(MODELS_DIR / "paddleOCR")
YOLO_MODELS_DIR = ensure_dir(MODELS_DIR / "yolo")

# Back-compat aliases (if older code imported these names)
TEMP_UPLOADED_DIR = UPLOAD_DIR
TEMP_PLATE_EXTRACTED_DIR = PLATES_DIR
