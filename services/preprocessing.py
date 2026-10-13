"""
Image Preprocessing Service
Provides rotation correction, deskewing, grayscale conversion,
denoising, contrast enhancement (CLAHE), adaptive thresholding,
and safe crop generation for field-level inspection.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

def load_image_cv(image_path: str) -> Optional[np.ndarray]:
    """Load image safely handling Unicode paths on Windows."""
    try:
        # cv2.imdecode handles Unicode paths safely
        with open(image_path, "rb") as f:
            bytes_data = bytearray(f.read())
            img_arr = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image_cv(image: np.ndarray, output_path: str) -> bool:
    """Save image safely handling Unicode paths on Windows."""
    try:
        ext = os.path.splitext(output_path)[1].lower() or ".jpg"
        if ext not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"
        success, encoded_img = cv2.imencode(ext, image)
        if success:
            with open(output_path, "wb") as f:
                f.write(encoded_img)
            return True
        return False
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False

def detect_skew_angle(image: np.ndarray) -> float:
    """
    Detect skew angle using minAreaRect on foreground text contours.
    Limits correction to +/- 45 degrees to avoid accidental 90 deg flips.
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Invert and threshold to find text components
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return 0.0

        angle = cv2.minAreaRect(coords)[-1]
        # In OpenCV, minAreaRect angle convention
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # If angle is negligible (<0.5 deg) or extreme (>20 deg), ignore to protect handwriting
        if abs(angle) < 0.5 or abs(angle) > 25.0:
            return 0.0
        return float(angle)
    except Exception:
        return 0.0

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around its center without cutting off edges."""
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Recompute new bounding box bounds
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_document_image(
    input_path: str,
    output_path: str,
    deskew: bool = True,
    enhance_contrast: bool = True,
    gentle_mode: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Apply gentle, non-destructive preprocessing pipeline:
    1. Deskew (gentle)
    2. Grayscale conversion
    3. Noise reduction (bilateral filter preserves pen stroke edges)
    4. Contrast enhancement (CLAHE)
    5. Adaptive thresholding (gentle, preserves thin pen lines)
    """
    meta = {
        "deskew_angle": 0.0,
        "width": 0,
        "height": 0,
        "channels": 3,
        "status": "success"
    }

    img = load_image_cv(input_path)
    if img is None:
        meta["status"] = "failed_to_load"
        return False, meta

    h, w = img.shape[:2]
    meta["width"] = w
    meta["height"] = h

    # 1. Gentle deskew
    if deskew:
        angle = detect_skew_angle(img)
        if abs(angle) > 0.5:
            img = rotate_image(img, angle)
            meta["deskew_angle"] = angle

    # 2. Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Bilateral filter for noise reduction (preserves stroke sharpness)
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)

    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
    else:
        enhanced = denoised

    # In gentle mode (default for handwriting), we preserve the high-detail enhanced grayscale image
    # rather than harsh binarization which fragments light handwriting or ballpoint pen strokes.
    processed = enhanced

    saved = save_image_cv(processed, output_path)
    if not saved:
        meta["status"] = "failed_to_save"
        return False, meta

    return True, meta

def crop_field_region(
    image_path: str,
    bbox: list,  # [ymin, xmin, ymax, xmax] in relative 0-1000 scale or absolute pixel values
    output_crop_path: str,
    padding_pct: float = 0.08
) -> Optional[str]:
    """
    Safely crop a region of interest for a field with contextual padding.
    Supports normalized [0, 1000] scale coordinates or absolute coordinates.
    """
    try:
        img = load_image_cv(image_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        ymin, xmin, ymax, xmax = bbox

        # Check if coordinates are in 0-1000 scale (standard AI vision format)
        if max(ymin, xmin, ymax, xmax) <= 1000:
            ymin_px = int((ymin / 1000.0) * h)
            ymax_px = int((ymax / 1000.0) * h)
            xmin_px = int((xmin / 1000.0) * w)
            xmax_px = int((xmax / 1000.0) * w)
        else:
            ymin_px, xmin_px, ymax_px, xmax_px = int(ymin), int(xmin), int(ymax), int(xmax)

        # Add contextual padding around handwriting
        pad_y = int((ymax_px - ymin_px) * padding_pct)
        pad_x = int((xmax_px - xmin_px) * padding_pct)

        y1 = max(0, ymin_px - pad_y)
        y2 = min(h, ymax_px + pad_y)
        x1 = max(0, xmin_px - pad_x)
        x2 = min(w, xmax_px + pad_x)

        if y2 <= y1 or x2 <= x1:
            return None

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if save_image_cv(crop, output_crop_path):
            return output_crop_path
        return None
    except Exception as e:
        print(f"Error cropping region: {e}")
        return None
