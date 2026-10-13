"""
Unified OCR Engine Module
Provides interchangeable OCR interfaces:
1. PyMuPDF Digital Text Engine (for native PDFs)
2. Tesseract OCR (with availability check and configuration feedback)
3. Gemini Vision Multimodal Engine (for handwriting, complex layouts, and vision)
4. Local Layout & Structural Parser
"""

import os
import re
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import fitz

# Check pytesseract
HAS_PYTESSERACT = False
try:
    import pytesseract
    from PIL import Image
    # Check if tesseract is actually executable
    try:
        pytesseract.get_tesseract_version()
        HAS_PYTESSERACT = True
    except Exception:
        HAS_PYTESSERACT = False
except Exception:
    HAS_PYTESSERACT = False

from services.gemini_vision import GeminiVisionClient

class OCREngine:
    def __init__(self, gemini_client: Optional[GeminiVisionClient] = None):
        self.gemini_client = gemini_client or GeminiVisionClient()
        self.has_tesseract = HAS_PYTESSERACT

    def get_engine_status(self) -> Dict[str, Any]:
        """Reports availability and health of all interchangeable OCR backends."""
        gemini_ok, gemini_err = self.gemini_client.is_available()
        return {
            "pymupdf": {
                "available": True,
                "description": "PyMuPDF text and layout extractor (active for digital PDFs)"
            },
            "gemini_vision": {
                "available": gemini_ok,
                "status": "ready" if gemini_ok else "unavailable",
                "message": gemini_err or "Gemini Multimodal Vision configured",
                "model": self.gemini_client.model_name
            },
            "tesseract": {
                "available": self.has_tesseract,
                "status": "ready" if self.has_tesseract else "not_installed_on_path",
                "message": "Tesseract binary found" if self.has_tesseract else "Tesseract OCR binary not found on PATH. Using PyMuPDF and Gemini/Local engines."
            }
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract native digital text from PDF."""
        text_parts = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                t = page.get_text()
                if t:
                    text_parts.append(t)
            doc.close()
        except Exception as e:
            print(f"Error reading PDF text: {e}")
        return "\n\n".join(text_parts).strip()

    def extract_text_with_tesseract(self, image_path: str) -> Tuple[bool, str, float]:
        """Runs Tesseract OCR if available; returns (success, text, avg_confidence)."""
        if not self.has_tesseract:
            return False, "", 0.0
        try:
            from PIL import Image
            img = Image.open(image_path)
            # Use image_to_data for confidence
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confs = [float(c) for c in data.get("conf", []) if str(c).replace(".", "").isdigit() and float(c) >= 0]
            avg_conf = sum(confs) / (len(confs) * 100.0) if confs else 0.0
            text = pytesseract.image_to_string(img)
            return True, text.strip(), avg_conf
        except Exception as e:
            print(f"Tesseract extraction error: {e}")
            return False, "", 0.0
