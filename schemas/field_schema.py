"""
Field Schema Definition
Defines the structure, validation status, and audit trail of an extracted field.
"""

from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field

class FieldExtractionResult(BaseModel):
    field_name: str
    display_name: Optional[str] = None
    value: Optional[str] = None
    confidence: float = 0.0
    source_type: str = "printed"  # printed, handwritten, machine_readable_zone, checkbox
    validation_status: str = "unreadable"  # valid, invalid, unreadable, missing
    needs_review: bool = True
    review_reason: Optional[str] = None
    original_prediction: Optional[str] = None
    reviewed_value: Optional[str] = None
    review_status: str = "pending"  # pending, confirmed, edited, rejected
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    raw_ocr_text: Optional[str] = None
    is_sensitive: bool = False
    mask_type: Optional[str] = None
    bounding_box: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "display_name": self.display_name or self.field_name,
            "value": self.value,
            "confidence": round(float(self.confidence), 4),
            "source_type": self.source_type,
            "validation_status": self.validation_status,
            "needs_review": self.needs_review,
            "review_reason": self.review_reason,
            "original_prediction": self.original_prediction,
            "reviewed_value": self.reviewed_value,
            "review_status": self.review_status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "raw_ocr_text": self.raw_ocr_text,
            "is_sensitive": self.is_sensitive,
            "bounding_box": self.bounding_box
        }

    def get_display_value(self, mask_sensitive: bool = True) -> str:
        """Returns the active value (reviewed if edited, else original) with masking if requested."""
        val = self.reviewed_value if self.review_status == "edited" and self.reviewed_value is not None else self.value
        if val is None or str(val).strip() == "":
            return "<Missing / Unreadable>"
        if not mask_sensitive or not self.is_sensitive:
            return str(val)

        val_str = str(val).strip()
        if self.mask_type == "aadhaar":
            # Mask all but last 4 digits
            digits = "".join(ch for ch in val_str if ch.isdigit())
            if len(digits) >= 4:
                return f"XXXX-XXXX-{digits[-4:]}"
            return "XXXX-XXXX-XXXX"
        elif self.mask_type == "pan":
            # Mask middle 4 numbers: ABCDE1234F -> ABCDE****F
            if len(val_str) == 10:
                return f"{val_str[:5]}****{val_str[-1]}"
            return "**********"
        elif self.mask_type == "account_number":
            if len(val_str) >= 4:
                return f"******{val_str[-4:]}"
            return "******"
        elif self.mask_type == "passport":
            if len(val_str) >= 3:
                return f"{val_str[:2]}*****{val_str[-1]}"
            return "********"
        return "********"
