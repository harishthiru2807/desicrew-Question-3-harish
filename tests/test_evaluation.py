"""
Unit tests for Ground-Truth Evaluation Service.
"""

import pytest
from schemas.document_schema import DocumentRecord, DocumentTypeInfo
from schemas.field_schema import FieldExtractionResult
from services.evaluation import EvaluationService

def test_ground_truth_evaluation():
    doc = DocumentRecord(
        document_id="DOC_001",
        filename="DOC_001_aadhaar_card.png",
        file_path="uploads/DOC_001/doc.png",
        document_type=DocumentTypeInfo(
            value="aadhaar_card",
            display_name="Aadhaar Card",
            confidence=0.95
        ),
        fields={
            "aadhaar_number": FieldExtractionResult(
                field_name="aadhaar_number",
                value="5482 9104 3821",
                confidence=0.92,
                source_type="printed",
                validation_status="valid",
                needs_review=False
            ),
            "full_name": FieldExtractionResult(
                field_name="full_name",
                value="Aarav Sharma",
                confidence=0.88,
                source_type="printed",
                validation_status="valid",
                needs_review=False
            )
        }
    )

    gt_data = {
        "DOC_001": {
            "document_id": "DOC_001",
            "document_type": "aadhaar_card",
            "fields": {
                "aadhaar_number": "5482 9104 3821",
                "full_name": "Aarav Sharma"
            }
        }
    }

    eval_res = EvaluationService.evaluate_batch([doc], gt_data)

    assert eval_res["docs_evaluated"] == 1
    assert eval_res["classification_accuracy_pct"] == 100.0
    assert eval_res["overall_field_accuracy_pct"] == 100.0
    assert eval_res["missing_field_rate_pct"] == 0.0
    assert eval_res["false_acceptance_rate_pct"] == 0.0
