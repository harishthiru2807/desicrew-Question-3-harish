"""
Unit tests for Confidence Scoring and Review Routing:
- Confidence score computation
- Threshold enforcement (0.85 default)
- Validation failure penalty
- Ambiguous date & checkbox penalty
"""

import pytest
from services.confidence import ConfidenceEngine

@pytest.fixture
def engine():
    return ConfidenceEngine(threshold=0.85)

def test_high_confidence_valid_field(engine):
    conf, needs_review, reason, breakdown = engine.compute_field_confidence(
        raw_model_confidence=0.92,
        validation_status="valid",
        source_type="printed",
        value="ABCPS1234F"
    )
    assert conf >= 0.85
    assert needs_review is False

def test_failed_validation_caps_confidence(engine):
    # Even if model reported 0.99, failed validation must penalize score and flag for review
    conf, needs_review, reason, breakdown = engine.compute_field_confidence(
        raw_model_confidence=0.99,
        validation_status="invalid",
        source_type="printed",
        value="INVALID_PAN",
        validation_error="Format mismatch"
    )
    assert conf <= 0.40
    assert needs_review is True
    assert "Validation failed" in reason

def test_ambiguous_date_flagged(engine):
    conf, needs_review, reason, breakdown = engine.compute_field_confidence(
        raw_model_confidence=0.90,
        validation_status="ambiguous_date",
        source_type="handwritten",
        value="03/04/2025"
    )
    assert conf < 0.85
    assert needs_review is True

def test_missing_value_zero_confidence(engine):
    conf, needs_review, reason, breakdown = engine.compute_field_confidence(
        raw_model_confidence=0.90,
        validation_status="missing",
        source_type="handwritten",
        value=None
    )
    assert conf == 0.0
    assert needs_review is True
