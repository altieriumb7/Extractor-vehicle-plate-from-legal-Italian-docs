# Insurance Vehicle Plate Verification

A Streamlit application for verifying vehicle license plates against user input using computer vision and OCR technology.

## Features

- License plate detection using YOLO object detection
- Plate alignment and image preprocessing
- OCR-based text recognition using PaddleOCR
- Text matching and verification against user input
- Interactive web interface with Streamlit
- Detailed verification reports


## Requirements

- Python 3.11
- Dependencies managed with Poetry

## Project Structure

```
├── app.py                  # Main Streamlit application
├── plate_detection.py      # License plate detection using YOLO
├── ocr_verification.py     # OCR and text verification using PaddleOCR
├── models/                 # Directory for model files
│   ├── yolo/               # YOLO model files
│   └── paddleOCR/          # PaddleOCR model files
├── temp_uploaded/          # Temporary directory for uploaded images
└── temp_plate_extracted/   # Temporary directory for extracted plates
```

## Installation

1. Make sure you have Poetry installed:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone this repository:
   ```bash
   git clone link.git
   cd insurance-plate-verification
   ```

3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

4. Download the required models:
   - YOLO model: Place in `models/yolo/best-seg-n-640-16epochs.pt`
   - PaddleOCR models: Place in `models/paddleOCR/` directory

## Usage

Run the application with:

```bash
poetry run streamlit run app.py
```



## How It Works

1. **Input Collection Stage**:
   - User uploads an image of a vehicle plate
   - User enters the expected vehicle plate 
   - Optional policy number can be provided

2. **Processing Stage**:
   - The application detects license plates using YOLO
   - Detected plates are extracted, aligned, and processed
   - OCR is performed to read text from the plates
   - Extracted text is compared with user-provided plate number

3. **Results Stage**:
   - Verification results are displayed with confidence scores
   - Detailed information is presented in a formatted table
   - Show detected plate if it has not been verified
   
## Performance

The application uses advanced computer vision techniques for optimal performance:
- Plate detection with YOLO segmentation models
- Rotation and alignment of plates by computer vision algorithms
- Extracting text by OCR 
- Text matching with similarity scoring

## Customization

You can adjust the following parameters in the code:
- OCR confidence thresholds
- Text similarity thresholds for matching
- Image preprocessing parameters

## License

Reply

## Contact

u.altieri@reply.it
