"""
Unit tests for JSON Serialization and Export Schema compliance.
"""

import json
import pytest
from schemas.document_schema import DocumentRecord, DocumentTypeInfo
from schemas.field_schema import FieldExtractionResult
from services.export_service import ExportService

def test_json_schema_export():
    doc = DocumentRecord(
        document_id="DOC_001",
        filename="test_pan.png",
        file_path="uploads/DOC_001/test_pan.png",
        document_type=DocumentTypeInfo(
            value="pan_card",
            display_name="PAN Card",
            confidence=0.96,
            classification_method="ocr_and_rules",
            classification_reasons=["Found Income Tax Department header"]
        ),
        fields={
            "pan_number": FieldExtractionResult(
                field_name="pan_number",
                value="ABCPS1234F",
                confidence=0.95,
                source_type="printed",
                validation_status="valid",
                needs_review=False,
                review_reason="High confidence and valid format",
                is_sensitive=True,
                mask_type="pan"
            )
        }
    )

    json_str = ExportService.export_single_document_json(doc)
    data = json.loads(json_str)

    assert data["document_id"] == "DOC_001"
    assert data["document_type"]["value"] == "pan_card"
    assert "pan_number" in data["fields"]
    assert data["fields"]["pan_number"]["value"] == "ABCPS1234F"
    assert data["fields"]["pan_number"]["confidence"] == 0.95
    assert data["document_review_status"] == "pending"

def test_flagging_report_generation():
    doc = DocumentRecord(
        document_id="DOC_001",
        filename="test_nach.png",
        file_path="uploads/DOC_001/test_nach.png",
        document_type=DocumentTypeInfo(
            value="nach_ecs_mandate",
            display_name="NACH / ECS Mandate",
            confidence=0.90,
            classification_method="ocr_and_rules"
        ),
        fields={
            "bank_account_number": FieldExtractionResult(
                field_name="bank_account_number",
                value="987654321012",
                confidence=0.88,
                source_type="handwritten",
                validation_status="valid",
                needs_review=False
            ),
            "ifsc_code": FieldExtractionResult(
                field_name="ifsc_code",
                value=None,
                confidence=0.0,
                source_type="handwritten",
                validation_status="missing",
                needs_review=True,
                review_reason="Field missing or unreadable in document"
            )
        }
    )

    df, metrics = ExportService.generate_flagging_report([doc], confidence_threshold=0.85)
    assert metrics["total_documents"] == 1
    assert metrics["total_fields_expected"] == 2
    assert metrics["total_fields_flagged"] == 1
    assert metrics["missing_fields"] == 1
    assert len(df) == 1
