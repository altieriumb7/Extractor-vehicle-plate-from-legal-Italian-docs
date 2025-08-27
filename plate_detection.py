import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile


class PlateDetector:
    def __init__(self, model_path):
        """
        Initialize the license plate detector with a YOLO model.

        Args:
            model_path (str): Path to the trained YOLO model
        """

        self.model = YOLO(model_path).eval()

    def detect_and_extract_plates(self, image, output_folder=None):
        """
        Detect and extract license plates from an image.

        Args:
            image: Input image as numpy array or path to image file
            output_folder (str, optional): Folder to save extracted plates, if None plates are not saved

        Returns:
            List of extracted plate images and their metadata
        """
        # Create temp folder for processing if output_folder is not provided
        if output_folder is None:
            temp_dir = tempfile.TemporaryDirectory()
            output_folder = temp_dir.name
        else:
            os.makedirs(output_folder, exist_ok=True)
            temp_dir = None

        # Handle both image path and numpy array
        if isinstance(image, str):
            img_path = image
            is_path = True
        else:
            # Save numpy array to temporary file
            temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_img_path = temp_img.name
            cv2.imwrite(temp_img_path, image)
            img_path = temp_img_path
            is_path = False

        # Run inference with the model
        results = self.model(img_path, verbose=False)

        extracted_plates = []

        # Check if we have any results
        if results and len(results) > 0:
            # Get the first result (usually there's only one per image)
            result = results[0]

            # Get the original image
            orig_img = result.orig_img

            # Extract masks if available
            if hasattr(result, 'masks') and result.masks is not None:
                # Get all masks
                masks = result.masks.data

                # Get boxes to associate masks with classes
                class_names = []
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    boxes_classes = result.boxes.cls
                    for cls_idx in boxes_classes:
                        class_idx = int(cls_idx.item())
                        class_names.append(result.names[class_idx])

                # Process each mask
                for i in range(len(masks)):
                    try:
                        # Get mask
                        mask = masks[i].cpu().numpy()

                        # Resize mask to match original image dimensions if needed
                        if mask.shape != orig_img.shape[:2]:
                            mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)

                        # Ensure mask is binary
                        binary_mask = (mask > 0.5).astype(np.uint8)

                        # Get class name if available, otherwise use index
                        class_name = class_names[i] if i < len(class_names) else f"class_{i}"

                        # Create a white background image
                        white_bg = np.ones_like(orig_img) * 255

                        # Apply mask: take pixels from original image where mask is true,
                        # and from white background where mask is false
                        masked_img = orig_img.copy()
                        for c in range(3):  # Apply to each color channel
                            masked_img[:, :, c] = orig_img[:, :, c] * binary_mask + white_bg[:, :, c] * (
                                        1 - binary_mask)

                        # Find bounding box of the mask to crop exactly to the license plate
                        y_indices, x_indices = np.where(binary_mask > 0)

                        if len(y_indices) > 0 and len(x_indices) > 0:
                            # Get bounds
                            x_min, x_max = np.min(x_indices), np.max(x_indices)
                            y_min, y_max = np.min(y_indices), np.max(y_indices)

                            # Crop the image to just the license plate area
                            cropped_plate = masked_img[y_min:y_max + 1, x_min:x_max + 1]

                            # Generate a unique filename if saving
                            if output_folder:
                                output_name = f"plate_{i}_{class_name}.jpg"
                                output_path = os.path.join(output_folder, output_name)

                                # Save the cropped masked image
                                cv2.imwrite(output_path, cropped_plate)

                            # Add to results list
                            plate_info = {
                                'plate_img': cropped_plate,
                                'class_name': class_name,
                                'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                                'mask': binary_mask,
                                'confidence': 1.0  # Placeholder, actual confidence could be retrieved from model
                            }
                            extracted_plates.append(plate_info)

                    except Exception as e:
                        print(f"Error processing mask {i}: {e}")

        # Clean up temporary files
        if not is_path:
            os.unlink(temp_img_path)

        if temp_dir:
            temp_dir.cleanup()

        return extracted_plates

    def rotate_and_align_plate(self, plate_image, threshold=240):
        """
        Rotate the license plate image to align it horizontally.

        Args:
            plate_image: Input plate image as numpy array
            threshold: Pixel value threshold for white

        Returns:
            Rotated and aligned plate image
        """
        # Convert to grayscale if not already
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # Threshold to get binary image (non-white pixels)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Find coordinates of non-white pixels
        non_white_pixels = np.where(binary > 0)

        # Check if there are enough non-white pixels
        if len(non_white_pixels[0]) < 10:
            return plate_image  # Return original if not enough points

        # Calculate covariance matrix of non-white pixel coordinates
        coords = np.column_stack([non_white_pixels[1], non_white_pixels[0]])  # [x, y] format

        # Use PCA to find the main axis of the non-white pixels
        mean = np.mean(coords, axis=0)
        coords_centered = coords - mean
        cov_matrix = np.cov(coords_centered.T)

        # Get eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Find index of largest eigenvalue
        largest_eigenvalue_idx = np.argmax(eigenvalues)
        largest_eigenvector = eigenvectors[:, largest_eigenvalue_idx]

        # Calculate angle in degrees
        angle = np.arctan2(largest_eigenvector[1], largest_eigenvector[0]) * 180 / np.pi

        # Adjust angle to make the major axis horizontal
        if angle < -45:
            angle = angle + 90
        elif angle > 45:
            angle = angle - 90

        # Get image dimensions
        (h, w) = plate_image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # Create rotation matrix
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

        # Apply affine transformation
        rotated = cv2.warpAffine(plate_image, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        return rotated
