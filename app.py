import streamlit as st
import os

import cv2
import numpy as np
import re
from PIL import Image
import datetime
import os, pathlib, requests, streamlit as st
from detector import PlateDetector
from verifier import PlateOCRVerifier


# Set up temporary directories
temp_uploaded_dir = "temp_uploaded"
temp_plate_extracted_dir = "temp_plate_extracted"

# Create directories if they don't exist
os.makedirs(temp_uploaded_dir, exist_ok=True)
os.makedirs(temp_plate_extracted_dir, exist_ok=True)

# Initialize session state variables
if "verification_stage" not in st.session_state:
    st.session_state.verification_stage = 1

# Import our custom modules
from plate_detection import PlateDetector
from ocr_verification import PlateOCRVerifier

# Set page configuration
st.set_page_config(
    page_title="Insurance Vehicle Plate Verification",
    page_icon="🔒",
    layout="wide"
)


# [CSS styling code remains the same]

# Initialize detection and OCR models

@st.cache_resource(show_spinner="Loading YOLO and OCR models…")
def load_models():
    out_path = pathlib.Path("/tmp/yolo_weights.pt")
    if not out_path.exists():
        url = st.secrets["WEIGHTS_RAW_URL"]
        headers = {"Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}"}
        st.info("Downloading YOLO weights from private repo…")
        r = requests.get(url, headers=headers, timeout=600)
        r.raise_for_status()
        out_path.write_bytes(r.content)

    detector = PlateDetector(str(out_path))
    verifier = PlateOCRVerifier()
    return detector, verifier

detector, verifier = load_models()



def collect_user_input():
    # Add custom CSS to reduce top spacing
    st.markdown("""
    <style>
    .custom-header-container {
        margin-top: -20px;
        padding-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Use the custom class on the container div
    st.markdown('<div class="card custom-header-container">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #005BAA; margin-top: 0;">Vehicle Plate Verification</h3>', unsafe_allow_html=True)

    # Create two columns for input with equal vertical alignment
    col1, col2 = st.columns(2)

    with col1:
        # Add matching paragraph tag for consistent styling and alignment
        st.markdown("<p><strong>Vehicle Plate:</strong></p>", unsafe_allow_html=True)
        # Vehicle plate input with empty label for alignment
        plate_number = st.text_input("",
                                     placeholder="e.g., AB123CD",
                                     key="plate_number_input",
                                     label_visibility="collapsed")

    with col2:
        # Image upload with consistent styling
        st.markdown("<p><strong>Upload Vehicle Document:</strong></p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("",
                                         type=["jpg", "jpeg", "png"],
                                         key="plate_image_upload",
                                         label_visibility="collapsed")

    # Close the card
    st.markdown('</div>', unsafe_allow_html=True)

    # Add a run button
    if st.button("Verify Registration"):
        # Validate input
        if not uploaded_file:
            st.error("Please upload an image of the vehicle plate.")
        elif not plate_number:
            st.error("Please enter a vehicle registration number.")
        else:
            # Clean temporary directories first
            clean_temp_directories()

            # Save the uploaded image to temp_uploaded directory
            temp_file_path = os.path.join(temp_uploaded_dir, "input_image.jpg")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Store inputs in session state
            st.session_state.plate_number = plate_number
            st.session_state.temp_file_path = temp_file_path

            # Advance to processing stage
            st.session_state.verification_stage = 2
            st.rerun()


def process_verification():
    # Add custom CSS to reduce column spacing
    st.markdown("""
    <style>
    /* Reduce padding between columns */
    .st-emotion-cache-1r6slb0 {
        gap: 0.5rem !important;
    }

    /* Make card more compact */
    .card {
        padding: 0.75rem !important;
    }

    /* Make section headers more compact */
    .card p strong {
        margin-bottom: 0.15rem;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create verification processing card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #005BAA; margin-top: 0;">Verification Processing</h3>', unsafe_allow_html=True)

    # Access inputs from session state
    plate_number = st.session_state.plate_number
    temp_file_path = st.session_state.temp_file_path

    # Create three columns with custom widths
    col1, col2, col3 = st.columns([6, 3, 2])

    # COLUMN 1: Original Image
    with col1:
        st.markdown("<p style='font-size: 18px;'><strong>Original Image</strong></p>", unsafe_allow_html=True)      # Display the uploaded image
        image = Image.open(temp_file_path)
        st.image(image, width=550)

        # Convert to OpenCV format for processing
        image_cv = np.array(image.convert('RGB'))
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Process the image
    with st.spinner("Detecting license plate..."):
        # 1. Use YOLO to detect license plates
        if detector:
            extracted_plates = detector.detect_and_extract_plates(temp_file_path,
                                                                  output_folder=temp_plate_extracted_dir)

        # Check if any plates were detected
        if not extracted_plates:
            st.warning("No license plates detected in the image.")

            # Add a New Verification button for cases with no plates detected
            if st.button("New Verification", key="no_plates_new_verification"):
                # Reset session state and go back to stage 1
                st.session_state.verification_stage = 1

                # Store keys to be deleted in session state
                for key in ['plate_number', 'ocr_results', 'match_found', 'best_match']:
                    if key in st.session_state:
                        del st.session_state[key]

                # Rerun first
                st.rerun()

            # Close the card and exit the function
            st.markdown('</div>', unsafe_allow_html=True)
            return

    # 2. Rotate and align extracted plates
    plate_paths = []
    for filename in os.listdir(temp_plate_extracted_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            plate_path = os.path.join(temp_plate_extracted_dir, filename)

            # Read plate image
            plate_img = cv2.imread(plate_path)
            if plate_img is not None:
                # Rotate and align
                aligned_plate = detector.rotate_and_align_plate(plate_img)
                # Save back the aligned version
                cv2.imwrite(plate_path, aligned_plate)
                plate_paths.append(plate_path)

    # 3. Perform OCR on each extracted plate
    ocr_results = []
    match_found = False
    best_match = None

    with st.spinner("Reading license plate text..."):
        for plate_path in plate_paths:
            # Process with PaddleOCR
            if verifier:
                result = verifier.process_image(plate_path, plate_number, similarity_threshold=0.93)

            result['plate_path'] = plate_path
            ocr_results.append(result)

            # Check if this is a match
            if result['is_match'] and (best_match is None or result['similarity'] > best_match['similarity']):
                best_match = result
                match_found = True

    # Store results in session state
    st.session_state.ocr_results = ocr_results
    st.session_state.match_found = match_found
    st.session_state.best_match = best_match

    # COLUMN 2: Extracted plates with OCR results below
    with col2:
        # Section 1: Extracted plates
        st.markdown("<p style='font-size: 18px;'><strong>Extracted & Aligned Plates</strong></p>",
                    unsafe_allow_html=True)
        # Display plates
        if plate_paths:
            if len(plate_paths) > 1:
                # For multiple plates, stack them vertically
                for i, plate_path in enumerate(plate_paths):
                    st.image(plate_path, caption=f"Plate {i + 1}", width=200)
            else:
                # Just one plate
                st.image(plate_paths[0], caption="Plate 1", width=200)

        # Section 2: OCR Results below plates
        st.markdown("<p style='font-size: 18px;'><strong>OCR Results</strong></p>", unsafe_allow_html=True)
        # Simplified display for OCR results
        for i, result in enumerate(ocr_results):
            status = "✅" if result['is_match'] else "❌"
            st.info(f"{status} {result['recognized_text']}")

    # COLUMN 3: Only Verification Results
    with col3:
        # Display verification results in the third column
        display_verification_results_colored(match_found, best_match, plate_number)

    # Close the card
    st.markdown('</div>', unsafe_allow_html=True)


def display_verification_results_colored(match_found, best_match, plate_number):
    """Display verification results with colored header based on verification status"""
    # Generate a report ID (kept for potential future use)
    report_id = f"VER-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Set header background color based on verification status
    header_bg_color = "#0A693E" if match_found and best_match else "#BE3A31"  # Green if verified, Red if not

    # More prominent verification header with dynamic background color
    st.markdown(f"""
    <div style="background-color: {header_bg_color}; color: white; padding: 10px; 
                border-radius: 3px; margin-bottom: 15px; text-align: center;
                font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 16px; font-weight: bold;">
      VERIFICATION RESULT
    </div>
    """, unsafe_allow_html=True)

    # Display user's plate number with adjusted spacing
    st.markdown(f"""
    <div style="background-color: #f0f0f0; border: 1px solid #ddd; padding: 10px; margin: 0 0 15px 0; 
                border-radius: 4px; text-align: center; font-size: 15px;
                font-family: 'Helvetica Neue', Arial, sans-serif;">
      <span>Plate in input: <strong style="font-size: 16px;">{plate_number}</strong></span>
    </div>
    """, unsafe_allow_html=True)

    # Display match status - adjusted spacing and font
    if match_found and best_match:
        similarity = best_match['similarity'] * 100
        st.markdown(
            f"""
            <div style="background-color: #EBF7ED; border: 2px solid #0A693E; padding: 12px; margin: 0 0 15px 0; 
                        border-radius: 4px; text-align: center; font-size: 16px; font-weight: bold;
                        font-family: 'Helvetica Neue', Arial, sans-serif;">
              ✅ VERIFIED: {similarity:.1f}%
            </div>
            """,
            unsafe_allow_html=True)
    else:
        # Not verified - show best OCR result if available
        # First display the NOT VERIFIED message
        st.markdown(
            f"""
            <div style="background-color: #FBEAE5; border: 2px solid #BE3A31; padding: 12px; margin: 0 0 15px 0; 
                        border-radius: 4px; text-align: center; font-size: 16px; font-weight: bold;
                        font-family: 'Helvetica Neue', Arial, sans-serif;">
              ⚠️ NOT VERIFIED
            </div>
            """,
            unsafe_allow_html=True)

        # Find the best OCR text
        best_ocr_text = ""
        if best_match:
            # Use the best match even though it wasn't good enough
            best_ocr_text = best_match['recognized_text']
        else:
            # Find the highest confidence OCR result from session state
            ocr_results = st.session_state.get('ocr_results', [])
            if ocr_results:
                # Find the result with highest confidence
                best_ocr_result = max(ocr_results, key=lambda x: x.get('confidence', 0))
                best_ocr_text = best_ocr_result.get('recognized_text', '')

        # Clean up the OCR text by removing specific patterns
        if best_ocr_text:
            # Only remove the specific patterns mentioned
            patterns_to_remove = ["(A)", "A)", "(A", ")"]

            # Replace each pattern with empty string
            for pattern in patterns_to_remove:
                best_ocr_text = best_ocr_text.replace(pattern, "")

            # Trim any extra whitespace
            best_ocr_text = best_ocr_text.strip()

            # Display the plate detected with improved spacing and font size
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border: 1px solid #ddd; padding: 10px; margin: 0 0 15px 0; 
                            border-radius: 4px; text-align: center; font-size: 15px;
                            font-family: 'Helvetica Neue', Arial, sans-serif;">
                  <span>Plate detected: <strong style="font-size: 16px;">{best_ocr_text}</strong></span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Add extra space before the button
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

    # Only the New Verification button remains
    if st.button("New Verification", key="new_verification_compact"):
        # Reset session state and go back to stage 1
        st.session_state.verification_stage = 1

        # Store keys to be deleted in session state
        for key in ['plate_number', 'ocr_results', 'match_found', 'best_match']:
            if key in st.session_state:
                del st.session_state[key]

        # Rerun first
        st.rerun()

def display_verification_results_compact():
    """Display more compact verification results for third column"""
    # Access results from session state
    plate_number = st.session_state.plate_number  # This is the user-entered plate
    match_found = st.session_state.match_found
    best_match = st.session_state.best_match

    # Generate a report ID (kept for potential future use)
    report_id = f"VER-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    # More prominent verification header with background
    st.markdown("""
    <div style="background-color: #0056b3; color: white; padding: 5px; border-radius: 3px; margin-bottom: 8px; text-align: center;">
      <strong>VERIFICATION RESULT</strong>
    </div>
    """, unsafe_allow_html=True)

    # Display user's plate number prominently
    st.markdown(f"""
    <div style="background-color: #f0f0f0; border: 1px solid #ddd; padding: 8px; margin: 5px 0; 
                border-radius: 4px; text-align: center; font-size: 1.1em;">
      <span>Plate in input: <strong>{plate_number}</strong></span>
    </div>
    """, unsafe_allow_html=True)

    # Display match status - more prominent
    if match_found and best_match:
        similarity = best_match['similarity'] * 100
        st.markdown(
            f"""
            <div style="background-color: #EBF7ED; border: 2px solid #0A693E; padding: 8px; margin: 5px 0; 
                        border-radius: 4px; font-weight: bold; text-align: center; font-size: 1.1em;">
              ✅ VERIFIED: {similarity:.1f}%
            </div>
            """,
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div style="background-color: #FBEAE5; border: 2px solid #BE3A31; padding: 8px; margin: 5px 0; 
                        border-radius: 4px; font-weight: bold; text-align: center; font-size: 1.1em;">
              ⚠️ NOT VERIFIED
            </div>
            """,
            unsafe_allow_html=True)

    # Only the New Verification button remains
    if st.button("New Verification", key="new_verification_compact"):
        # Reset session state and go back to stage 1
        st.session_state.verification_stage = 1

        # Store keys to be deleted in session state
        for key in ['plate_number', 'ocr_results', 'match_found', 'best_match']:
            if key in st.session_state:
                del st.session_state[key]

        # Rerun first
        st.rerun()


def clean_temp_directories():
    """Clean up temporary directories"""
    # Clean temp_uploaded directory
    for filename in os.listdir(temp_uploaded_dir):
        file_path = os.path.join(temp_uploaded_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error cleaning up {file_path}: {str(e)}")

    # Clean temp_plate_extracted directory
    for filename in os.listdir(temp_plate_extracted_dir):
        file_path = os.path.join(temp_plate_extracted_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error cleaning up {file_path}: {str(e)}")


# Determine which stage to display
if st.session_state.verification_stage == 1:
    # Clean up temporary files when starting a new verification
    # This ensures we don't have old files lingering from previous runs
    clean_temp_directories()
    collect_user_input()
elif st.session_state.verification_stage == 2:
    process_verification()
