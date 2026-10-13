"""
Confidence Scoring and Review Routing Engine
Calculates calibrated field-level confidence scores combining:
1. Model-reported / OCR recognition confidence (0.0 to 1.0)
2. Field-specific structural and checksum validation results
3. Handwriting vs printed source-type uncertainty factor
4. Format ambiguity penalties (e.g. date ambiguity, multiple checkboxes)

Provides transparent scoring justification and routes fields to human review based
on a configurable threshold (default 0.85).
"""

from typing import Tuple, Dict, Any, Optional

DEFAULT_THRESHOLD = 0.85

THRESHOLD_JUSTIFICATION = (
    "The default operational threshold of 0.85 is selected as a high-integrity baseline for "
    "financial and identity workflows (KYC, banking mandates, policy declarations). A 0.85 cutoff "
    "mandates human review whenever model certainty is suboptimal or when field validation encounters "
    "ambiguities (e.g. cursive stroke degradation, ambiguous DD/MM ordering, or multiple checkbox fills). "
    "It is an operational safety threshold subject to calibration against ground truth, not an arbitrary optimum."
)

class ConfidenceEngine:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def compute_field_confidence(
        self,
        raw_model_confidence: Optional[float],
        validation_status: str,  # 'valid', 'valid_with_warning', 'ambiguous_date', 'ambiguous_checkbox', 'invalid', 'missing'
        source_type: str,        # 'printed', 'handwritten', 'machine_readable_zone', 'checkbox'
        value: Optional[str],
        validation_error: Optional[str] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        Computes composite confidence score and determines if field needs human review.

        Returns:
            (calibrated_confidence, needs_review, review_reason, score_breakdown)
        """
        breakdown = {
            "model_reported_confidence": raw_model_confidence,
            "source_type": source_type,
            "validation_status": validation_status,
            "adjustments": []
        }

        # 1. Base recognition score
        if raw_model_confidence is not None:
            base_score = float(raw_model_confidence)
        else:
            # Fallback heuristic: 0.70 for printed if recognized, 0.50 for handwritten
            base_score = 0.70 if source_type == "printed" else 0.50
            breakdown["adjustments"].append("OCR confidence unavailable; applied conservative base heuristic")

        # Missing or unreadable values automatically get 0.0 confidence and need review
        if value is None or str(value).strip() == "" or validation_status == "missing":
            return 0.0, True, "Field missing or unreadable in document", breakdown

        score = base_score

        # 2. Validation status impact (A high confidence score MUST NOT override failed validation!)
        if validation_status == "invalid":
            score = min(score, 0.40)
            breakdown["adjustments"].append(f"Validation failed ({validation_error}); capped confidence at 0.40")
            reason = f"Validation failed: {validation_error or 'Format invalid'}"
            return round(score, 4), True, reason, breakdown

        elif validation_status == "ambiguous_date":
            score = min(score, 0.65)
            breakdown["adjustments"].append("Ambiguous date order (DD/MM vs MM/DD); penalized confidence to 0.65")
            reason = validation_error or "Ambiguous date order requires human confirmation"
            return round(score, 4), True, reason, breakdown

        elif validation_status == "ambiguous_checkbox":
            score = min(score, 0.50)
            breakdown["adjustments"].append("Multiple or faint checkbox selections detected; penalized confidence to 0.50")
            reason = validation_error or "Ambiguous checkbox selection"
            return round(score, 4), True, reason, breakdown

        elif validation_status == "valid_with_warning":
            score = min(score, 0.75)
            breakdown["adjustments"].append(f"Validation warning: {validation_error}")
            reason = validation_error or "Validation warning flagged for review"
            return round(score, 4), True, reason, breakdown

        elif validation_status == "valid":
            # Passing strict format validation increases reliability
            if source_type in ["machine_readable_zone", "printed"]:
                score = min(1.0, score + 0.05)
                breakdown["adjustments"].append("Strict format match bonus (+0.05)")
            elif source_type == "handwritten":
                # Handwritten fields retain slight margin of uncertainty
                score = min(0.95, score)

        # 3. Final review threshold evaluation
        score = max(0.0, min(1.0, score))
        needs_review = score < self.threshold
        
        if needs_review:
            reason = f"Confidence {score:.2f} is below operational threshold {self.threshold:.2f}"
        else:
            reason = "High confidence and valid format"

        return round(score, 4), needs_review, reason, breakdown
