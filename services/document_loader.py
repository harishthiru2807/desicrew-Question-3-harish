"""
Document Loader Service
Handles file ingestion, batch validation, PDF rendering to images using PyMuPDF,
and assignment of unique document identifiers.
"""

import os
import shutil
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

from config import UPLOAD_DIR, PROCESSED_DIR
from services.preprocessing import preprocess_document_image

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

class DocumentLoader:
    def __init__(self, upload_dir: Path = UPLOAD_DIR, processed_dir: Path = PROCESSED_DIR):
        self.upload_dir = Path(upload_dir)
        self.processed_dir = Path(processed_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_document(self, file_bytes: bytes, filename: str, doc_index: int = 1) -> Dict[str, Any]:
        """
        Ingests an uploaded document, saves the raw file, renders pages to images,
        and applies preprocessing.
        """
        doc_id = f"DOC_{doc_index:03d}"
        ext = os.path.splitext(filename)[1].lower()
        
        result: Dict[str, Any] = {
            "document_id": doc_id,
            "filename": filename,
            "file_path": "",
            "extension": ext,
            "page_count": 0,
            "image_paths": [],           # Preprocessed image paths
            "original_image_paths": [],  # Original RGB image paths
            "digital_text": "",          # Embedded text if PDF has it
            "status": "pending",
            "error": None
        }

        if ext not in SUPPORTED_EXTENSIONS:
            result["status"] = "unsupported_format"
            result["error"] = f"Unsupported file extension: {ext}. Expected PDF, PNG, JPG, or JPEG."
            return result

        # Save uploaded file
        doc_storage_dir = self.upload_dir / doc_id
        doc_storage_dir.mkdir(parents=True, exist_ok=True)
        raw_file_path = doc_storage_dir / filename
        
        try:
            with open(raw_file_path, "wb") as f:
                f.write(file_bytes)
            result["file_path"] = str(raw_file_path)
        except Exception as e:
            result["status"] = "storage_failed"
            result["error"] = f"Failed to save file: {str(e)}"
            return result

        # Process PDF vs Image
        try:
            if ext == ".pdf":
                self._process_pdf(raw_file_path, doc_id, result)
            else:
                self._process_single_image(raw_file_path, doc_id, result)
                
            if result["page_count"] == 0 or not result["original_image_paths"]:
                result["status"] = "empty_or_corrupt"
                result["error"] = "No valid pages or image data could be extracted."
            else:
                result["status"] = "loaded"
        except Exception as e:
            result["status"] = "corrupt_file"
            result["error"] = f"Error reading document: {str(e)}"

        return result

    def _process_pdf(self, pdf_path: Path, doc_id: str, result: Dict[str, Any]):
        """Render each PDF page to an image and extract digital text."""
        doc = fitz.open(str(pdf_path))
        result["page_count"] = len(doc)
        
        all_digital_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                all_digital_text.append(text)

            # Render page at 200 DPI for optimal OCR clarity without ballooning memory
            pix = page.get_pixmap(dpi=200)
            orig_page_path = self.upload_dir / doc_id / f"page_{page_num + 1}_orig.png"
            pix.save(str(orig_page_path))
            result["original_image_paths"].append(str(orig_page_path))

            # Preprocess page image
            proc_page_path = self.processed_dir / f"{doc_id}_page_{page_num + 1}_proc.png"
            success, _ = preprocess_document_image(
                input_path=str(orig_page_path),
                output_path=str(proc_page_path),
                deskew=True,
                enhance_contrast=True
            )
            if success:
                result["image_paths"].append(str(proc_page_path))
            else:
                result["image_paths"].append(str(orig_page_path))

        result["digital_text"] = "\n\n".join(all_digital_text).strip()
        doc.close()

    def _process_single_image(self, img_path: Path, doc_id: str, result: Dict[str, Any]):
        """Process a single image file (JPG, PNG, JPEG)."""
        # Validate readability with PIL
        with Image.open(str(img_path)) as img:
            img.verify()
            
        result["page_count"] = 1
        orig_page_path = self.upload_dir / doc_id / "page_1_orig.png"
        
        # Convert and save as standardized PNG
        with Image.open(str(img_path)) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(str(orig_page_path), "PNG")
            
        result["original_image_paths"].append(str(orig_page_path))

        # Preprocess
        proc_page_path = self.processed_dir / f"{doc_id}_page_1_proc.png"
        success, _ = preprocess_document_image(
            input_path=str(orig_page_path),
            output_path=str(proc_page_path),
            deskew=True,
            enhance_contrast=True
        )
        if success:
            result["image_paths"].append(str(proc_page_path))
        else:
            result["image_paths"].append(str(orig_page_path))

    def clear_session_data(self):
        """Clears temporary session uploads and processed artifacts for privacy."""
        try:
            for d in [self.upload_dir, self.processed_dir]:
                if d.exists():
                    for item in d.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                        else:
                            item.unlink(missing_ok=True)
            return True
        except Exception as e:
            print(f"Error clearing session data: {e}")
            return False
