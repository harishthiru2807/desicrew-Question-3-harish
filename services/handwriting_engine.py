"""
Handwriting Recognition and Checkbox Extraction Engine
Specialized pipeline for handwritten insurance forms, NACH mandates, and declarations.
Handles field localization, contextual crop generation, stroke-preserving preprocessing,
checkbox contour analysis, and candidate reading comparison.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from services.preprocessing import crop_field_region, save_image_cv, load_image_cv
from services.gemini_vision import GeminiVisionClient

class HandwritingEngine:
    def __init__(self, gemini_client: Optional[GeminiVisionClient] = None):
        self.gemini_client = gemini_client or GeminiVisionClient()

    def process_handwritten_crop(
        self,
        image_path: str,
        bbox: list,
        field_name: str,
        output_crop_dir: Path
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates and saves a clean, stroke-preserved crop for visual audit and review.
        Returns: (orig_crop_path, enhanced_crop_path)
        """
        output_crop_dir.mkdir(parents=True, exist_ok=True)
        crop_filename = f"crop_{field_name}.png"
        orig_crop_path = str(output_crop_dir / crop_filename)

        actual_crop = crop_field_region(image_path, bbox, orig_crop_path, padding_pct=0.10)
        if not actual_crop:
            return None, None

        # Create gentle contrast-enhanced copy preserving light ink strokes
        enhanced_crop_path = str(output_crop_dir / f"enhanced_{field_name}.png")
        crop_img = load_image_cv(actual_crop)
        if crop_img is not None:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6))
            enhanced = clahe.apply(gray)
            save_image_cv(enhanced, enhanced_crop_path)
        else:
            enhanced_crop_path = orig_crop_path

        return orig_crop_path, enhanced_crop_path

    def analyze_checkboxes_opencv(
        self,
        image_path: str,
        options_catalog: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        OpenCV contour analysis for form checkboxes:
        Detects square contours of standard checkbox dimensions (e.g. 15-40px),
        measures internal pixel fill density, and detects ticked/crossed boxes.
        """
        img = load_image_cv(image_path)
        if img is None:
            return {
                "detected": False,
                "selected_reason": None,
                "is_ambiguous": True,
                "reason": "Unable to load image for checkbox analysis"
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binary inverse threshold
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h)
            # Checkbox characteristics: roughly square, width 12 to 50 pixels
            if 0.8 <= aspect_ratio <= 1.25 and 12 <= w <= 55 and 12 <= h <= 55:
                # Measure internal fill (exclude 2px border)
                if w > 6 and h > 6:
                    roi = thresh[y+3:y+h-3, x+3:x+w-3]
                    fill_ratio = np.count_nonzero(roi) / float(roi.size)
                    candidate_boxes.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "fill_ratio": fill_ratio,
                        "is_checked": fill_ratio > 0.18  # marked with tick or cross or fill
                    })

        # Sort candidate checkboxes vertically (top to bottom)
        candidate_boxes = sorted(candidate_boxes, key=lambda b: b["y"])
        checked_boxes = [b for b in candidate_boxes if b["is_checked"]]

        default_reasons = options_catalog or [
            "Tax planning / Investment growth",
            "Risk coverage / Life protection",
            "Family financial security",
            "Retirement planning",
            "Loan liability cover"
        ]

        if len(checked_boxes) == 1:
            idx = candidate_boxes.index(checked_boxes[0])
            reason_text = default_reasons[idx] if idx < len(default_reasons) else f"Option {idx + 1}"
            return {
                "detected": True,
                "selected_reason": reason_text,
                "is_ambiguous": False,
                "fill_ratio": checked_boxes[0]["fill_ratio"],
                "box_count": len(candidate_boxes),
                "checked_count": 1
            }
        elif len(checked_boxes) > 1:
            return {
                "detected": True,
                "selected_reason": "Multiple options selected",
                "is_ambiguous": True,
                "reason": f"Detected {len(checked_boxes)} marked checkboxes; form expects a single choice",
                "box_count": len(candidate_boxes),
                "checked_count": len(checked_boxes)
            }
        else:
            return {
                "detected": len(candidate_boxes) > 0,
                "selected_reason": None,
                "is_ambiguous": True,
                "reason": "No clear checkbox mark detected",
                "box_count": len(candidate_boxes),
                "checked_count": 0
            }
