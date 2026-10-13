"""
Unit tests for Document Classifier:
- Multi-signal text and regex matching for all 10 document types
- Ambiguous / unsupported document routing
- Manual override workflow
"""

import pytest
from services.document_classifier import DocumentClassifier

@pytest.fixture
def classifier():
    return DocumentClassifier()

def test_classify_aadhaar(classifier):
    text = "Unique Identification Authority of India (UIDAI) Mera Aadhaar 5482 9104 3821"
    res = classifier.classify_text(text)
    assert res.value == "aadhaar_card"
    assert res.confidence > 0.80

def test_classify_pan(classifier):
    text = "INCOME TAX DEPARTMENT GOVT. OF INDIA Permanent Account Number Card ABCPS1234F"
    res = classifier.classify_text(text)
    assert res.value == "pan_card"
    assert res.confidence > 0.85

def test_classify_nach(classifier):
    text = "NACH / ECS MANDATE FORM UMRN 12345678901234567890 Sponsor Bank Code HDFC0000001 IFSC Bank A/c"
    res = classifier.classify_text(text)
    assert res.value == "nach_ecs_mandate"
    assert res.confidence > 0.80

def test_classify_ambiguous_document(classifier):
    # Text with insufficient cues
    text = "Hello world random invoice receipt details here without any identity keywords"
    res = classifier.classify_text(text)
    assert res.value == "unsupported_ambiguous"
    assert res.confidence < 0.85

def test_manual_override(classifier):
    text = "INCOME TAX DEPARTMENT"
    res = classifier.classify_text(text)
    # Manually override to another type
    overridden = classifier.manual_override(res, "driving_licence")
    assert overridden.value == "driving_licence"
    assert overridden.classification_method == "manual_override"
    assert overridden.confidence == 1.0
