# app.py

# --- Streamlit config MUST come first ---
import streamlit as st
st.set_page_config(
    page_title="Insurance Vehicle Plate Verification",
    page_icon="🔒",
    layout="wide"
)

# --- Stdlib / third-party imports ---
import os
import pathlib
from pathlib import Path
import datetime

import cv2
import numpy as np
from PIL import Image
import requests

# --- Local modules ---
# Keep a single, consistent import source for your classes
from detector import PlateDetector           # <-- your repo's module
from verifier import PlateOCRVerifier       # <-- your repo's module

# Use the temp-dir utilities (no writes into the repo!)
from utils_paths import get_runtime_dirs, cleanup_runtime_dirs, models_root


# ---------------------------
# Paths & Session Directories
# ---------------------------
APP_DIR = Path(__file__).parent
DIRS = get_runtime_dirs(st.session_state)   # {'base', 'uploaded', 'plates'}

UPLOAD_DIR: Path = DIRS["uploaded"]
PLATES_DIR: Path = DIRS["plates"]
MODELS_DIR: Path = models_root(APP_DIR)

# Ensure temp dirs exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PLATES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize stage
if "verification_stage" not in st.session_state:
    st.session_state.verification_stage = 1


# ---------------------------
# Model loading (cached)
# ---------------------------
@st.cache_resource(show_spinner="Loading YOLO and OCR models…")
def load_models():
    """
    - Pull YOLO weights from a private URL (via GitHub token from st.secrets).
    - Store them in system temp (read+write allowed).
    - Instantiate your detector and OCR verifier.
    """
    out_path = Path("/tmp/yolo_weights.pt")
    if not out_path.exists():
        if "WEIGHTS_RAW_URL" not in st.secrets or "GITHUB_TOKEN" not in st.secrets:
            st.error("Missing WEIGHTS_RAW_URL or GITHUB_TOKEN in Streamlit secrets.")
            st.stop()

        url = st.secrets["WEIGHTS_RAW_URL"]
        headers = {"Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}"}
        st.info("Downloading YOLO weights from private repo…")
        r = requests.get(url, headers=headers, timeout=600)
        r.raise_for_status()
        out_path.write_bytes(r.content)

    detector = PlateDetector(str(out_path))  # your class should accept path to weights
    verifier = PlateOCRVerifier()
    return detector, verifier


detector, verifier = load_models()


# ---------------------------
# Helpers
# ---------------------------
def clean_temp_directories():
    """Clean up session-scoped temp directories (uploaded + plates)."""
    for p in [UPLOAD_DIR, PLATES_DIR]:
        if not p.exists():
            continue
        for child in p.iterdir():
            try:
                if child.is_file():
                    child.unlink(missing_ok=True)
            except Exception as e:
                st.error(f"Error cleaning {child}: {e}")


def save_uploaded_file(uploaded_file) -> Path:
    """
    Persist the uploaded file into the session's temp upload dir.
    Returns the saved path (Path object).
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        # We only support images reliably. PDFs require pdf2image/poppler.
        # If you need PDF support, add pdf2image + poppler and convert here.
        st.error("Please upload an image (.jpg, .jpeg, .png). PDF is not supported in this build.")
        st.stop()

    target = UPLOAD_DIR / f"input_image{suffix}"
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def display_verification_results_colored(match_found, best_match, plate_number):
    """Display verification results with colored header based on verification status"""
    # report_id kept for potential future use
    _ = f"VER-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    header_bg_color = "#0A693E" if match_found and best_match else "#BE3A31"  # Green if verified, Red if not

    st.markdown(
        f"""
        <div style="background-color: {header_bg_color}; color: white; padding: 10px; 
                    border-radius: 3px; margin-bottom: 15px; text-align: center;
                    font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 16px; font-weight: bold;">
          VERIFICATION RESULT
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="background-color: #f0f0f0; border: 1px solid #ddd; padding: 10px; margin: 0 0 15px 0; 
                    border-radius: 4px; text-align: center; font-size: 15px;
                    font-family: 'Helvetica Neue', Arial, sans-serif;">
          <span>Plate in input: <strong style="font-size: 16px;">{plate_number}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if match_found and best_match:
        similarity = best_match.get("similarity", 0.0) * 100
        st.markdown(
            f"""
            <div style="background-color: #EBF7ED; border: 2px solid #0A693E; padding: 12px; margin: 0 0 15px 0; 
                        border-radius: 4px; text-align: center; font-size: 16px; font-weight: bold;
                        font-family: 'Helvetica Neue', Arial, sans-serif;">
              ✅ VERIFIED: {similarity:.1f}%
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color: #FBEAE5; border: 2px solid #BE3A31; padding: 12px; margin: 0 0 15px 0; 
                        border-radius: 4px; text-align: center; font-size: 16px; font-weight: bold;
                        font-family: 'Helvetica Neue', Arial, sans-serif;">
              ⚠️ NOT VERIFIED
            </div>
            """,
            unsafe_allow_html=True,
        )

        best_ocr_text = ""
        if best_match:
            best_ocr_text = best_match.get("recognized_text", "") or ""

        if not best_ocr_text:
            ocr_results = st.session_state.get("ocr_results", [])
            if ocr_results:
                best_ocr_result = max(ocr_results, key=lambda x: x.get("confidence", 0))
                best_ocr_text = best_ocr_result.get("recognized_text", "") or ""

        if best_ocr_text:
            for pattern in ["(A)", "A)", "(A", ")"]:
                best_ocr_text = best_ocr_text.replace(pattern, "")
            best_ocr_text = best_ocr_text.strip()

            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border: 1px solid #ddd; padding: 10px; margin: 0 0 15px 0; 
                            border-radius: 4px; text-align: center; font-size: 15px;
                            font-family: 'Helvetica Neue', Arial, sans-serif;">
                  <span>Plate detected: <strong style="font-size: 16px;">{best_ocr_text}</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    if st.button("New Verification", key="new_verification_compact"):
        st.session_state.verification_stage = 1
        for key in ["plate_number", "ocr_results", "match_found", "best_match", "temp_file_path"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()


# ---------------------------
# Stage 1: Collect Input
# ---------------------------
def collect_user_input():
    st.markdown(
        """
        <style>
        .custom-header-container { margin-top: -20px; padding-top: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card custom-header-container">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #005BAA; margin-top: 0;">Vehicle Plate Verification</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<p><strong>Vehicle Plate:</strong></p>", unsafe_allow_html=True)
        plate_number = st.text_input(
            "", placeholder="e.g., AB123CD", key="plate_number_input", label_visibility="collapsed"
        )

    with col2:
        st.markdown("<p><strong>Upload Vehicle Document (image only):</strong></p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Run button
    if st.button("Verify Registration"):
        if uploaded_file is None:
            st.error("Please upload an image of the vehicle plate.")
            return
        if not plate_number:
            st.error("Please enter a vehicle registration number.")
            return

        # Reset workspace & persist upload
        clean_temp_directories()
        saved_path = save_uploaded_file(uploaded_file)

        # Store inputs for stage 2
        st.session_state.plate_number = plate_number.strip().upper()
        st.session_state.temp_file_path = str(saved_path)

        st.session_state.verification_stage = 2
        st.experimental_rerun()


# ---------------------------
# Stage 2: Process Verification
# ---------------------------
def process_verification():
    st.markdown(
        """
        <style>
        .card { padding: 0.75rem !important; }
        .card p strong { margin-bottom: 0.15rem; display: block; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #005BAA; margin-top: 0;">Verification Processing</h3>', unsafe_allow_html=True)

    # Inputs from stage 1
    plate_number = st.session_state.get("plate_number", "")
    temp_file_path = st.session_state.get("temp_file_path", "")

    if not plate_number or not temp_file_path:
        st.error("Missing input. Please start a new verification.")
        st.session_state.verification_stage = 1
        st.experimental_rerun()

    col1, col2, col3 = st.columns([6, 3, 2])

    # COLUMN 1: Original Image
    with col1:
        st.markdown("<p style='font-size: 18px;'><strong>Original Image</strong></p>", unsafe_allow_html=True)
        image = Image.open(temp_file_path).convert("RGB")
        st.image(image, width=550)

        # Convert to OpenCV format if needed in future steps
        image_cv = np.array(image)
        _ = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Detect + extract plates
    with st.spinner("Detecting license plate..."):
        extracted_plates = []
        if detector:
            # Your detector should take the input path and write crops into PLATES_DIR
            extracted_plates = detector.detect_and_extract_plates(
                temp_file_path, output_folder=str(PLATES_DIR)
            )

        if not extracted_plates:
            st.warning("No license plates detected in the image.")
            if st.button("New Verification", key="no_plates_new_verification"):
                st.session_state.verification_stage = 1
                for key in ["plate_number", "ocr_results", "match_found", "best_match", "temp_file_path"]:
                    st.session_state.pop(key, None)
                st.experimental_rerun()

            st.markdown("</div>", unsafe_allow_html=True)
            return

    # Rotate & align extracted plates (in-place)
    plate_paths = []
    for p in PLATES_DIR.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            plate_img = cv2.imread(str(p))
            if plate_img is None:
                continue
            aligned = detector.rotate_and_align_plate(plate_img)
            cv2.imwrite(str(p), aligned)
            plate_paths.append(str(p))

    # OCR
    ocr_results = []
    match_found = False
    best_match = None

    with st.spinner("Reading license plate text..."):
        for plate_path in plate_paths:
            if verifier:
                result = verifier.process_image(
                    plate_path, plate_number, similarity_threshold=0.93
                )
            else:
                result = {"is_match": False, "recognized_text": "", "similarity": 0.0, "confidence": 0.0}

            result["plate_path"] = plate_path
            ocr_results.append(result)

            if result.get("is_match"):
                if best_match is None or result.get("similarity", 0) > best_match.get("similarity", 0):
                    best_match = result
                    match_found = True

    # Persist results in session
    st.session_state.ocr_results = ocr_results
    st.session_state.match_found = match_found
    st.session_state.best_match = best_match

    # COLUMN 2: Extracted plates + OCR results
    with col2:
        st.markdown("<p style='font-size: 18px;'><strong>Extracted & Aligned Plates</strong></p>", unsafe_allow_html=True)
        if plate_paths:
            if len(plate_paths) > 1:
                for i, pth in enumerate(plate_paths):
                    st.image(pth, caption=f"Plate {i+1}", width=200)
            else:
                st.image(plate_paths[0], caption="Plate 1", width=200)

        st.markdown("<p style='font-size: 18px;'><strong>OCR Results</strong></p>", unsafe_allow_html=True)
        for i, res in enumerate(ocr_results):
            status = "✅" if res.get("is_match") else "❌"
            st.info(f"{status} {res.get('recognized_text', '')}")

    # COLUMN 3: Final verdict
    with col3:
        display_verification_results_colored(match_found, best_match, plate_number)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Main Flow
# ---------------------------
if st.session_state.verification_stage == 1:
    clean_temp_directories()
    collect_user_input()
elif st.session_state.verification_stage == 2:
    process_verification()
