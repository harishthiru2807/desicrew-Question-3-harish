"""
Field Extraction Engine
Orchestrates document-specific field extraction, rule-based label matching,
MRZ decoding, checkbox integration, field validation, and confidence scoring.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from config import DOCUMENT_TYPES, get_document_fields
from schemas.field_schema import FieldExtractionResult
from services.validators import FieldValidator
from services.confidence import ConfidenceEngine
from services.handwriting_engine import HandwritingEngine

class FieldExtractor:
    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_engine = ConfidenceEngine(threshold=confidence_threshold)
        self.validator = FieldValidator()
        self.handwriting_engine = HandwritingEngine()

    def set_threshold(self, threshold: float):
        self.confidence_engine.threshold = threshold

    def extract_document_fields(
        self,
        doc_type: str,
        text: str,
        vision_result: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None,
        crop_dir: Optional[Path] = None
    ) -> Dict[str, FieldExtractionResult]:
        """
        Extracts all required fields for the identified document type.
        Combines Vision API results (if available) with local label-value pattern matchers.
        """
        required_fields = get_document_fields(doc_type)
        extracted_results: Dict[str, FieldExtractionResult] = {}

        # 1. Start with Vision API results if provided
        vision_fields = vision_result.get("fields", {}) if vision_result else {}

        for f_def in required_fields:
            fname = f_def["field_name"]
            disp_name = f_def.get("display_name", fname)
            stype = f_def.get("source_type", "printed")
            is_sens = f_def.get("is_sensitive", False)
            mask_type = f_def.get("mask_type")

            raw_val = None
            raw_ocr = None
            model_conf = None
            bbox = None

            if fname in vision_fields:
                v_data = vision_fields[fname]
                raw_val = v_data.get("value")
                raw_ocr = v_data.get("raw_text") or raw_val
                model_conf = v_data.get("confidence")
                bbox = v_data.get("bounding_box")

            # 2. Local Regex and Label Extraction Fallback / Verification
            local_val, local_conf = self._extract_field_locally(doc_type, fname, text, image_path)

            if raw_val is None or str(raw_val).strip() == "" or str(raw_val).lower() == "null":
                raw_val = local_val
                raw_ocr = local_val
                model_conf = local_conf

            # 3. Field-specific Validation
            is_valid, val_status, val_err, normalized_val = self._validate_field(doc_type, fname, raw_val)

            active_val = normalized_val if is_valid and normalized_val is not None else raw_val

            # 4. Calibrated Confidence Scoring & Review Routing
            conf_score, needs_rev, rev_reason, _ = self.confidence_engine.compute_field_confidence(
                raw_model_confidence=model_conf,
                validation_status=val_status,
                source_type=stype,
                value=active_val,
                validation_error=val_err
            )

            # 5. Build Result
            result_obj = FieldExtractionResult(
                field_name=fname,
                display_name=disp_name,
                value=active_val,
                confidence=conf_score,
                source_type=stype,
                validation_status=val_status,
                needs_review=needs_rev,
                review_reason=rev_reason,
                original_prediction=str(raw_val) if raw_val is not None else None,
                reviewed_value=None,
                review_status="pending",
                raw_ocr_text=str(raw_ocr) if raw_ocr is not None else None,
                is_sensitive=is_sens,
                mask_type=mask_type,
                bounding_box=bbox
            )

            extracted_results[fname] = result_obj

        return extracted_results

    def _validate_field(self, doc_type: str, field_name: str, value: Optional[str]):
        """Routes value to appropriate domain validator."""
        if value is None or str(value).strip() == "":
            return False, "missing", "Field is missing or empty", None

        # Aadhaar
        if field_name == "aadhaar_number":
            return self.validator.validate_aadhaar_number(value)
        # PAN
        elif field_name in ["pan_number", "tin_pan"]:
            return self.validator.validate_pan_number(value)
        # DL
        elif field_name == "dl_number":
            return self.validator.validate_dl_number(value)
        # Passport
        elif field_name == "passport_number":
            return self.validator.validate_passport_number(value)
        elif field_name == "mrz_line_2":
            return self.validator.validate_mrz_line_2(value)
        # Banking
        elif field_name == "ifsc_code":
            return self.validator.validate_ifsc_code(value)
        elif field_name == "bank_account_number":
            return self.validator.validate_bank_account_number(value)
        elif field_name == "amount_in_figures":
            return self.validator.validate_amount(value)
        elif field_name == "frequency":
            return self.validator.validate_frequency(value)
        # Dates
        elif "date" in field_name or field_name in ["date_of_birth", "date_of_issue", "valid_till_date", "date_of_expiry"]:
            return self.validator.validate_date(value)
        # Checkbox
        elif field_name == "reason_for_multiple_policies":
            return self.validator.validate_checkbox(value)
        # Text fields
        else:
            return self.validator.validate_generic_text(value, field_label=field_name.replace("_", " ").title())

    def _extract_field_locally(
        self,
        doc_type: str,
        field_name: str,
        text: str,
        image_path: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[float]]:
        """Extracts field using targeted regexes and layout anchors."""
        if not text:
            return None, None

        # 1. Aadhaar Card
        if doc_type == "aadhaar_card":
            if field_name == "aadhaar_number":
                # Look for 12 digits (with optional spaces: 4 4 4)
                m = re.search(r"\b([2-9]\d{3}\s\d{4}\s\d{4})\b", text)
                if m:
                    return m.group(1), 0.90
                m_no_space = re.search(r"\b([2-9]\d{11})\b", text)
                if m_no_space:
                    digits = m_no_space.group(1)
                    return f"{digits[:4]} {digits[4:8]} {digits[8:]}", 0.85
            elif field_name == "date_of_birth":
                m = re.search(r"(?:DOB|Date of Birth|Year of Birth)[:\s]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4}|[12][90]\d{2})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.88
            elif field_name == "full_name":
                m = re.search(r"(?:Name[:\s]*|Government of India\n)([A-Z][a-zA-Z\s]{3,30})\n", text)
                if m:
                    return m.group(1).strip(), 0.78
            elif field_name == "address":
                m = re.search(r"(?:Address|To)[:\s]*\n?([\s\S]{10,120}?\d{6})", text, re.IGNORECASE)
                if m:
                    clean_addr = re.sub(r"\s+", " ", m.group(1)).strip()
                    return clean_addr, 0.80

        # 2. PAN Card
        elif doc_type == "pan_card":
            if field_name == "pan_number":
                m = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", text)
                if m:
                    return m.group(1), 0.95
            elif field_name == "full_name":
                m = re.search(r"(?:Name|NAME)[:\s]*\n?([A-Z\s]{3,35})\n", text)
                if m:
                    return m.group(1).strip(), 0.82
            elif field_name == "fathers_name":
                m = re.search(r"(?:Father's Name|FATHER'S NAME)[:\s]*\n?([A-Z\s]{3,35})\n", text)
                if m:
                    return m.group(1).strip(), 0.82
            elif field_name == "date_of_birth":
                m = re.search(r"(?:Date of Birth|DATE OF BIRTH)[:\s]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})", text)
                if m:
                    return m.group(1), 0.90

        # 3. Driving Licence
        elif doc_type == "driving_licence":
            if field_name == "dl_number":
                m = re.search(r"\b([A-Z]{2}[0-9]{2}[ -]?[0-9]{4}[0-9]{7}|[A-Z]{2}[0-9]{2}[ -]?[0-9]{11})\b", text)
                if m:
                    return m.group(1).replace(" ", ""), 0.92
            elif field_name == "name":
                m = re.search(r"(?:Name|Holder Name)[:\s]*([A-Z][a-zA-Z\s]{3,30})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80
            elif field_name == "date_of_issue":
                m = re.search(r"(?:Date of Issue|DOI|Issued on)[:\s]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85
            elif field_name == "valid_till_date":
                m = re.search(r"(?:Valid Till|Validity|Expiry)[:\s]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85

        # 4. Passport
        elif doc_type == "passport":
            if field_name == "passport_number":
                m = re.search(r"\b([A-PR-WYa-pr-wy][1-9][0-9]{6})\b", text)
                if m:
                    return m.group(1), 0.94
            elif field_name == "date_of_birth":
                m = re.search(r"(?:Date of Birth|DOB)[:\s]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.88
            elif field_name == "date_of_expiry":
                m = re.search(r"(?:Date of Expiry|Expiry Date)[:\s]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.88
            elif field_name == "mrz_line_2":
                # Look for 44 character line containing passport digits, DOB, Expiry
                lines = [line.strip().replace(" ", "") for line in text.split("\n")]
                for l in lines:
                    if len(l) == 44 and "<" in l and not l.startswith("P<"):
                        return l, 0.92

        # 5. NACH / ECS Mandate
        elif doc_type == "nach_ecs_mandate":
            if field_name == "ifsc_code":
                m = re.search(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", text)
                if m:
                    return m.group(1), 0.90
            elif field_name == "bank_account_number":
                m = re.search(r"(?:Bank A/c No|Account Number|A/c No)[:\s]*([0-9]{9,18})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85
                # Fallback digit sequence
                seqs = re.findall(r"\b([0-9]{11,16})\b", text)
                if seqs:
                    return seqs[0], 0.70
            elif field_name == "bank_name":
                m = re.search(r"(?:Bank Name|with Bank)[:\s]*([A-Za-z\s]{4,30}(?:Bank|Ltd|Limited))", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80
            elif field_name == "amount_in_figures":
                m = re.search(r"(?:Amount|₹|Rs\.?)[:\s]*([0-9,]+(?:\.[0-9]{2})?)", text, re.IGNORECASE)
                if m:
                    return m.group(1).replace(",", ""), 0.80
            elif field_name == "frequency":
                m = re.search(r"\b(Monthly|Quarterly|Half-Yearly|Yearly|As and when presented)\b", text, re.IGNORECASE)
                if m:
                    return m.group(1).title(), 0.85

        # 6. FATCA Annexure Form
        elif doc_type == "fatca_annexure_form":
            if field_name == "policy_number":
                m = re.search(r"(?:Policy No|Policy Number)[:\s]*([A-Z0-9]{6,16})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85
            elif field_name == "tin_pan":
                m = re.search(r"(?:TIN|PAN|Tax Identification Number)[:\s]*([A-Z]{5}[0-9]{4}[A-Z]|[A-Z0-9]{8,12})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.88
            elif field_name == "fathers_name":
                m = re.search(r"(?:Father's Name|Father Name)[:\s]*([A-Za-z\s]{3,30})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80
            elif field_name == "place_of_birth":
                m = re.search(r"(?:Place of Birth|City of Birth)[:\s]*([A-Za-z\s]{3,25})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80
            elif field_name == "nationality":
                m = re.search(r"(?:Nationality|Country of Citizenship)[:\s]*([A-Za-z]{3,20})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.85

        # 7. Benefit Illustration Declaration
        elif doc_type == "benefit_illustration_declaration":
            if field_name == "application_number":
                m = re.search(r"(?:Application No|Proposal No)[:\s]*([A-Z0-9]{6,16})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85
            elif field_name == "policyholder_name":
                m = re.search(r"(?:Policyholder Name|Name of Policyholder)[:\s]*([A-Za-z\s]{3,30})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80
            elif field_name == "date":
                m = re.search(r"(?:Date|Dated)[:\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{4})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85
            elif field_name == "place":
                m = re.search(r"(?:Place|Location)[:\s]*([A-Za-z\s]{3,25})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80

        # 8. Moral Hazard Questionnaire
        elif doc_type == "moral_hazard_questionnaire":
            if field_name == "application_number":
                m = re.search(r"(?:Application No|Proposal No)[:\s]*([A-Z0-9]{6,16})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85
            elif field_name == "name_of_life_assured":
                m = re.search(r"(?:Life Assured|Name of Life Assured)[:\s]*([A-Za-z\s]{3,30})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80
            elif field_name == "nominee_relationship":
                m = re.search(r"(?:Relationship with Life Assured|Nominee Relationship)[:\s]*([A-Za-z\s]{3,20})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.85
            elif field_name == "date":
                m = re.search(r"(?:Date|Dated)[:\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{4})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85
            elif field_name == "place":
                m = re.search(r"(?:Place|Location)[:\s]*([A-Za-z\s]{3,25})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80

        # 9. Multiple Policies Consent Form
        elif doc_type == "multiple_policies_consent_form":
            if field_name == "proposer_name":
                m = re.search(r"(?:Proposer Name|Name of Proposer)[:\s]*([A-Za-z\s]{3,30})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80
            elif field_name == "reason_for_multiple_policies":
                # If image is available, run OpenCV checkbox detector
                if image_path:
                    cb_res = self.handwriting_engine.analyze_checkboxes_opencv(image_path)
                    if cb_res.get("detected") and cb_res.get("selected_reason"):
                        conf = 0.85 if not cb_res.get("is_ambiguous") else 0.50
                        return cb_res["selected_reason"], conf
                # Fallback text search for common reason phrases
                for r in ["Tax planning", "Risk coverage / Life protection", "Family financial security", "Retirement planning"]:
                    if r.lower() in text.lower():
                        return r, 0.75
            elif field_name == "date":
                m = re.search(r"(?:Date|Dated)[:\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{4})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85
            elif field_name == "place":
                m = re.search(r"(?:Place|Location)[:\s]*([A-Za-z\s]{3,25})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80

        # 10. Suitability Profiler Declaration
        elif doc_type == "suitability_profiler_declaration":
            if field_name == "application_number":
                m = re.search(r"(?:Application No|Proposal No)[:\s]*([A-Z0-9]{6,16})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85
            elif field_name == "name_of_life_assured":
                m = re.search(r"(?:Name of Life Assured|Customer Name)[:\s]*([A-Za-z\s]{3,30})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80
            elif field_name == "name_of_agent_sp":
                m = re.search(r"(?:Agent/SP Name|Name of Agent|Specified Person)[:\s]*([A-Za-z\s]{3,30})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80
            elif field_name == "date":
                m = re.search(r"(?:Date|Dated)[:\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{4})", text, re.IGNORECASE)
                if m:
                    return m.group(1), 0.85
            elif field_name == "place":
                m = re.search(r"(?:Place|Location)[:\s]*([A-Za-z\s]{3,25})", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip(), 0.80

        return None, None
