import difflib
import re

from paddleocr import PaddleOCR
det_model_path = "models/paddleOCR/det"  # Path to detection model
rec_model_path = "models/paddleOCR/rec"  # Path to recognition model
cls_model_path = "models/paddleOCR/cls"  # Optional: Path to classification model

class PlateOCRVerifier:
    def __init__(self):
        """
        Initialize the OCR verifier with PaddleOCR.
        Args:
            use_angle_cls (bool): Whether to use angle classifier for OCR
            lang (str): Language to use for OCR model
        """
        self.ocr = PaddleOCR(
            use_angle_cls=False,  # Set to True if you want to use the classification model
            lang='en',
            det_model_dir=det_model_path,
            rec_model_dir=rec_model_path,
            cls_model_dir=cls_model_path
        )

    def perform_ocr(self, image):
        """
        Perform OCR on the given image.

        Args:
            image: Input image as numpy array or path to image file

        Returns:
            Tuple of (recognized_text, confidence)
        """
        try:
            # Handle both image path and numpy array
            if isinstance(image, str):
                # Image is already a file path
                img_path = image
            else:
                # Image is a numpy array, we'll use it directly
                img_path = image

            # Run OCR on the image
            ocr_result = self.ocr.ocr(img_path)

            # Extract text from results
            texts = []
            if ocr_result and len(ocr_result) > 0:
                for line in ocr_result:
                    if line:
                        for word_info in line:
                            if len(word_info) >= 2:
                                text = word_info[1][0]  # The actual text
                                conf = word_info[1][1]  # Confidence score
                                texts.append((text, conf))

            # Process OCR results
            if texts:
                recognized_text = " ".join([t[0] for t in texts])
                confidence = sum([t[1] for t in texts]) / len(texts)
                return recognized_text, confidence
            else:
                return "", 0.0

        except Exception as e:
            return "", 0.0


    def verify_plate(self, detected_text, expected_plate, similarity_threshold=0.9):
        """
        Compare detected plate text with expected plate number.

        Args:
            detected_text (str): Text detected by OCR
            expected_plate (str): Expected plate number provided by user
            similarity_threshold (float): Threshold for considering a match (0-1)

        Returns:
            Tuple of (is_match, similarity_score)
        """
        if not detected_text or not expected_plate:
            return False, 0.0

        # Clean and standardize both texts
        detected_clean = re.sub(r'[^A-Z0-9]', '', detected_text.upper())
        expected_clean = re.sub(r'[^A-Z0-9]', '', expected_plate.upper())

        # Calculate similarity using sequence matcher
        similarity = difflib.SequenceMatcher(None, detected_clean, expected_clean).ratio()

        # Determine if it's a match based on threshold
        is_match = similarity >= similarity_threshold
        return is_match, similarity

    def process_image(self, image, expected_plate, similarity_threshold=0.9):
        """
        Complete processing pipeline: perform OCR and verify against expected plate.

        Args:
            image: Input image (numpy array or path)
            expected_plate (str): Expected plate number
            similarity_threshold (float): Threshold for match determination

        Returns:
            Dict with OCR and verification results
        """
        # Perform OCR
        recognized_text, confidence = self.perform_ocr(image)
        # Verify against expected plate
        is_match, similarity = self.verify_plate(
            recognized_text, expected_plate, similarity_threshold
        )

        # Return complete results
        return {
            'recognized_text': recognized_text,
            'confidence': confidence,
            'expected_plate': expected_plate,
            'is_match': is_match,
            'similarity': similarity,
            'match_status': "MATCH" if is_match else "NO_MATCH" if recognized_text else "NO_TEXT_DETECTED"
        }
