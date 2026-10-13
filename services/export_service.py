"""
Export Service
Generates structured JSON and CSV reports conforming strictly to evaluation schema:
1. Individual Document JSON
2. Batch JSON
3. Human-Review Report CSV
4. Field-Level Extraction Report CSV
5. Consolidated Flagging Report CSV & Summary Metrics
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from schemas.document_schema import DocumentRecord
from config import EXPORTS_DIR

class ExportService:
    @staticmethod
    def export_single_document_json(doc: DocumentRecord, output_path: Optional[Path] = None) -> str:
        """Serializes single document to JSON matching Section 11 schema."""
        data = doc.to_export_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @staticmethod
    def export_batch_json(docs: List[DocumentRecord], output_path: Optional[Path] = None) -> str:
        """Serializes full document batch to combined JSON."""
        data = [d.to_export_dict() for d in docs]
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    FIELD_COLUMNS = [
        "Document ID", "Filename", "Document Type", "Field Name", "Raw Field Key",
        "Extracted Value", "Raw Unmasked Value", "Confidence", "Source Type",
        "Validation Status", "Needs Review", "Review Reason", "Original Prediction",
        "Reviewed Value", "Review Status", "Reviewed By", "Reviewed At"
    ]

    FLAGGING_COLUMNS = [
        "Document ID", "Filename", "Document Type", "Field Name",
        "Extracted Value", "Confidence Score", "Threshold", "Validation Status",
        "Review Reason", "Source Type", "Review Status"
    ]

    @staticmethod
    def generate_field_extraction_dataframe(docs: List[DocumentRecord], mask_sensitive: bool = True) -> pd.DataFrame:
        """Returns a Pandas DataFrame of all extracted fields across documents."""
        rows = []
        for d in docs:
            for fname, f in d.fields.items():
                rows.append({
                    "Document ID": d.document_id,
                    "Filename": d.filename,
                    "Document Type": d.document_type.display_name,
                    "Field Name": f.display_name or fname,
                    "Raw Field Key": fname,
                    "Extracted Value": f.get_display_value(mask_sensitive=mask_sensitive),
                    "Raw Unmasked Value": f.value,
                    "Confidence": f.confidence,
                    "Source Type": f.source_type,
                    "Validation Status": f.validation_status,
                    "Needs Review": "Yes" if f.needs_review else "No",
                    "Review Reason": f.review_reason,
                    "Original Prediction": f.original_prediction,
                    "Reviewed Value": f.reviewed_value,
                    "Review Status": f.review_status,
                    "Reviewed By": f.reviewed_by,
                    "Reviewed At": f.reviewed_at
                })
        if not rows:
            return pd.DataFrame(columns=ExportService.FIELD_COLUMNS)
        return pd.DataFrame(rows)

    @staticmethod
    def generate_flagging_report(
        docs: List[DocumentRecord],
        confidence_threshold: float,
        mask_sensitive: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generates consolidated flagging report table and actual calculated metrics.
        """
        rows = []
        total_docs = len(docs)
        classified_success = sum(1 for d in docs if d.document_type.value != "unsupported_ambiguous")
        classification_failures = total_docs - classified_success
        total_fields_expected = sum(len(d.fields) for d in docs)
        total_fields_extracted = 0
        total_fields_flagged = 0
        missing_fields = 0
        invalid_fields = 0
        handwritten_flagged = 0

        for d in docs:
            for fname, f in d.fields.items():
                has_val = f.value is not None and str(f.value).strip() != ""
                if has_val:
                    total_fields_extracted += 1
                else:
                    missing_fields += 1

                if f.validation_status == "invalid":
                    invalid_fields += 1

                if f.needs_review:
                    total_fields_flagged += 1
                    if f.source_type == "handwritten":
                        handwritten_flagged += 1

                    rows.append({
                        "Document ID": d.document_id,
                        "Filename": d.filename,
                        "Document Type": d.document_type.display_name,
                        "Field Name": f.display_name or fname,
                        "Extracted Value": f.get_display_value(mask_sensitive=mask_sensitive),
                        "Confidence Score": f.confidence,
                        "Threshold": confidence_threshold,
                        "Validation Status": f.validation_status,
                        "Review Reason": f.review_reason or "Low confidence / validation warning",
                        "Source Type": f.source_type,
                        "Review Status": f.review_status
                    })

        metrics = {
            "total_documents": total_docs,
            "classified_successfully": classified_success,
            "classification_failures": classification_failures,
            "total_fields_expected": total_fields_expected,
            "total_fields_extracted": total_fields_extracted,
            "total_fields_flagged": total_fields_flagged,
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields,
            "handwritten_fields_flagged": handwritten_flagged,
            "extraction_rate_pct": round((total_fields_extracted / total_fields_expected * 100), 1) if total_fields_expected > 0 else 0.0,
            "flagging_rate_pct": round((total_fields_flagged / total_fields_expected * 100), 1) if total_fields_expected > 0 else 0.0
        }

        if not rows:
            df_flagged = pd.DataFrame(columns=ExportService.FLAGGING_COLUMNS)
        else:
            df_flagged = pd.DataFrame(rows)
        return df_flagged, metrics
