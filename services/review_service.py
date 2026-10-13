"""
Review Service
Manages human-in-the-loop audit logs, corrections, field approvals,
and document-level status transitions. Preserves original machine predictions.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from schemas.document_schema import DocumentRecord
from schemas.field_schema import FieldExtractionResult

class ReviewService:
    @staticmethod
    def confirm_field(
        doc_record: DocumentRecord,
        field_name: str,
        reviewer_id: str = "human_reviewer"
    ) -> bool:
        """Confirms an extracted field value as verified."""
        if field_name not in doc_record.fields:
            return False
        f = doc_record.fields[field_name]
        f.review_status = "confirmed"
        f.reviewed_value = f.value
        f.reviewed_by = reviewer_id
        f.reviewed_at = datetime.now().isoformat()
        f.needs_review = False
        f.review_reason = "Confirmed by human reviewer"
        ReviewService._update_doc_status(doc_record)
        return True

    @staticmethod
    def edit_field_value(
        doc_record: DocumentRecord,
        field_name: str,
        new_value: str,
        reviewer_id: str = "human_reviewer"
    ) -> bool:
        """Edits an extracted value, preserving original machine prediction."""
        if field_name not in doc_record.fields:
            return False
        f = doc_record.fields[field_name]
        if f.original_prediction is None:
            f.original_prediction = f.value
        f.reviewed_value = new_value.strip()
        f.review_status = "edited"
        f.reviewed_by = reviewer_id
        f.reviewed_at = datetime.now().isoformat()
        f.needs_review = False
        f.review_reason = "Edited and approved by human reviewer"
        ReviewService._update_doc_status(doc_record)
        return True

    @staticmethod
    def mark_field_unreadable(
        doc_record: DocumentRecord,
        field_name: str,
        reviewer_id: str = "human_reviewer"
    ) -> bool:
        """Marks a field as verified unreadable."""
        if field_name not in doc_record.fields:
            return False
        f = doc_record.fields[field_name]
        if f.original_prediction is None:
            f.original_prediction = f.value
        f.reviewed_value = None
        f.validation_status = "unreadable"
        f.review_status = "rejected"
        f.reviewed_by = reviewer_id
        f.reviewed_at = datetime.now().isoformat()
        f.needs_review = False
        f.review_reason = "Marked unreadable by human reviewer"
        ReviewService._update_doc_status(doc_record)
        return True

    @staticmethod
    def confirm_all_fields(
        doc_record: DocumentRecord,
        reviewer_id: str = "human_reviewer"
    ) -> int:
        """Confirms all remaining pending fields in a document."""
        confirmed_count = 0
        for f in doc_record.fields.values():
            if f.review_status == "pending":
                f.review_status = "confirmed"
                f.reviewed_value = f.value
                f.reviewed_by = reviewer_id
                f.reviewed_at = datetime.now().isoformat()
                f.needs_review = False
                f.review_reason = "Batch confirmed by reviewer"
                confirmed_count += 1
        ReviewService._update_doc_status(doc_record)
        return confirmed_count

    @staticmethod
    def _update_doc_status(doc_record: DocumentRecord):
        """Sets document review status to 'reviewed' if no fields need review."""
        if all(not f.needs_review for f in doc_record.fields.values()):
            doc_record.document_review_status = "reviewed"
        else:
            doc_record.document_review_status = "pending"
