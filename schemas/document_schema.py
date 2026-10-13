"""
Document Schema Definition
Defines the structure for processed documents, classification metadata, and batch summaries.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from schemas.field_schema import FieldExtractionResult

class DocumentTypeInfo(BaseModel):
    value: str  # e.g., "aadhaar_card"
    display_name: str  # e.g., "Aadhaar Card"
    confidence: float = 0.0
    classification_method: str = "ai_multimodal_vision"  # ai_multimodal_vision, ocr_and_rules, manual_override
    original_predicted_type: Optional[str] = None
    classification_reasons: List[str] = Field(default_factory=list)

class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    file_path: str
    page_count: int = 1
    image_paths: List[str] = Field(default_factory=list)  # Preprocessed images
    original_image_paths: List[str] = Field(default_factory=list)
    document_type: DocumentTypeInfo
    fields: Dict[str, FieldExtractionResult] = Field(default_factory=dict)
    document_review_status: str = "pending"  # pending, reviewed, completed
    processing_time_seconds: float = 0.0
    processing_error: Optional[str] = None
    is_ambiguous_or_unsupported: bool = False

    def to_export_dict(self) -> Dict[str, Any]:
        """Convert to the standard export JSON format specified in Project Objective."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "document_type": {
                "value": self.document_type.value,
                "display_name": self.document_type.display_name,
                "confidence": round(float(self.document_type.confidence), 4),
                "classification_method": self.document_type.classification_method,
                "original_predicted_type": self.document_type.original_predicted_type,
                "reasons": self.document_type.classification_reasons
            },
            "fields": {
                fname: fresult.to_dict() for fname, fresult in self.fields.items()
            },
            "document_review_status": self.document_review_status,
            "processing_time_seconds": round(float(self.processing_time_seconds), 2),
            "is_ambiguous_or_unsupported": self.is_ambiguous_or_unsupported
        }

    def get_flagged_fields_count(self) -> int:
        return sum(1 for f in self.fields.values() if f.needs_review)

    def get_handwritten_flagged_count(self) -> int:
        return sum(1 for f in self.fields.values() if f.source_type == "handwritten" and f.needs_review)
