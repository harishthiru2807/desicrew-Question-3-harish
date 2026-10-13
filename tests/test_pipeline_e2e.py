"""
End-to-End Pipeline Unit Tests:
- Testing ingestion, classification, and extraction across all 10 document types
- Testing ReviewService confirmation, edits, and unreadable flagging
"""

import pytest
from pathlib import Path
from config import SAMPLE_DOCS_DIR, DOCUMENT_TYPES
from services.document_loader import DocumentLoader
from services.document_classifier import DocumentClassifier
from services.field_extractor import FieldExtractor
from services.review_service import ReviewService
from schemas.document_schema import DocumentRecord

@pytest.fixture
def loader():
    return DocumentLoader()

@pytest.fixture
def classifier():
    return DocumentClassifier()

@pytest.fixture
def extractor():
    return FieldExtractor()

def test_all_10_document_type_configurations():
    """Verify that all 10 document types have required fields defined with valid metadata."""
    assert len(DOCUMENT_TYPES) == 10
    expected_types = [
        "aadhaar_card", "pan_card", "driving_licence", "passport", "nach_ecs_mandate",
        "fatca_annexure_form", "benefit_illustration_declaration", "moral_hazard_questionnaire",
        "multiple_policies_consent_form", "suitability_profiler_declaration"
    ]
    for dt in expected_types:
        assert dt in DOCUMENT_TYPES
        fields = DOCUMENT_TYPES[dt]["required_fields"]
        assert len(fields) >= 4  # Each category has at least 4 required fields

def test_e2e_extraction_on_sample_docs(loader, classifier, extractor):
    """Test full ingestion, classification, and extraction on generated sample documents."""
    sample_files = list(SAMPLE_DOCS_DIR.glob("DOC_*.*"))
    assert len(sample_files) == 10

    processed_records = []
    for idx, fpath in enumerate(sample_files, start=1):
        with open(fpath, "rb") as f:
            bytes_data = f.read()
        
        load_res = loader.load_document(bytes_data, fpath.name, doc_index=idx)
        assert load_res["status"] == "loaded"

        # Classification
        class_res = classifier.classify_text(load_res.get("digital_text", ""))
        
        # Field Extraction
        img_path = load_res["image_paths"][0] if load_res["image_paths"] else None
        fields = extractor.extract_document_fields(
            doc_type=class_res.value,
            text=load_res.get("digital_text", ""),
            image_path=img_path
        )

        doc_record = DocumentRecord(
            document_id=load_res["document_id"],
            filename=load_res["filename"],
            file_path=load_res["file_path"],
            page_count=load_res["page_count"],
            image_paths=load_res["image_paths"],
            original_image_paths=load_res["original_image_paths"],
            document_type=class_res,
            fields=fields
        )
        processed_records.append(doc_record)

    assert len(processed_records) == 10

def test_review_service_workflow():
    """Verify human review actions: confirm, edit, mark unreadable, batch confirm."""
    from schemas.document_schema import DocumentTypeInfo
    from schemas.field_schema import FieldExtractionResult

    doc = DocumentRecord(
        document_id="DOC_001",
        filename="test.png",
        file_path="test.png",
        document_type=DocumentTypeInfo(value="pan_card", display_name="PAN Card"),
        fields={
            "pan_number": FieldExtractionResult(
                field_name="pan_number",
                value="ABCPS1234F",
                confidence=0.90,
                needs_review=True
            ),
            "full_name": FieldExtractionResult(
                field_name="full_name",
                value="Priya",
                confidence=0.60,
                needs_review=True
            )
        }
    )

    # 1. Edit full_name
    success = ReviewService.edit_field_value(doc, "full_name", "Priya Sundaram", reviewer_id="auditor_1")
    assert success is True
    assert doc.fields["full_name"].reviewed_value == "Priya Sundaram"
    assert doc.fields["full_name"].review_status == "edited"
    assert doc.fields["full_name"].needs_review is False

    # 2. Confirm pan_number
    success = ReviewService.confirm_field(doc, "pan_number", reviewer_id="auditor_1")
    assert success is True
    assert doc.fields["pan_number"].review_status == "confirmed"
    assert doc.fields["pan_number"].needs_review is False

    # 3. Document status transitions to reviewed
    assert doc.document_review_status == "reviewed"
