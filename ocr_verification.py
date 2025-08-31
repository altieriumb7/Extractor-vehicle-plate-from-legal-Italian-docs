from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
from paddleocr import PaddleOCR


@st.cache_resource
def get_ocr() -> PaddleOCR:
    """Create a single cached PaddleOCR instance.
    No local model directories are specified; default models are auto-downloaded
    to the user's cache on first run.
    """
    # Change to 'it' if you specifically want Italian-trained models.
    lang = os.getenv("PADDLE_LANG", "en")
    # Use defaults so Streamlit Cloud can fetch models automatically.
    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)


def _parse_ocr_result(result: List[Any]) -> Tuple[str, float, List[str], List[float]]:
    """Flatten PaddleOCR's nested result into text, avg confidence, lines, confs."""
    lines: List[str] = []
    confs: List[float] = []
    for page in (result or []):
        if not page:
            continue
        for det in page:
            # det: [box, (text, conf)]
            try:
                text = det[1][0]
                conf = float(det[1][1])
            except Exception:
                continue
            if text:
                lines.append(str(text).strip())
                confs.append(conf)
    full_text = " ".join(lines).strip()
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return full_text, avg_conf, lines, confs


def ocr_image(image_path: str) -> Dict[str, Any]:
    """Run OCR on an image file path and return a structured dict.

    Returns: {"text": str, "avg_conf": float, "lines": List[str], "confidences": List[float]}
    """
    ocr = get_ocr()
    result = ocr.ocr(image_path, cls=True)
    text, avg_conf, lines, confs = _parse_ocr_result(result)
    return {"text": text, "avg_conf": avg_conf, "lines": lines, "confidences": confs}


# Optional convenience wrapper kept for backward compatibility
# (if your app previously called a function named `extract_text`).

def extract_text(image_path: str) -> Dict[str, Any]:
    return ocr_image(image_path)
