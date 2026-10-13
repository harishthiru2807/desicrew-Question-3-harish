"""
Document Classification Service
Multi-signal classifier integrating:
- Keyword density & weighted phrase matching
- Structural identifier regex presence (PAN, Aadhaar, MRZ, UMRN/IFSC)
- Layout headers and form titles
- AI Vision verification
- Explicit detection of ambiguous and unsupported documents
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from config import DOCUMENT_TYPES, get_document_type_names
from schemas.document_schema import DocumentTypeInfo

# Weighted signature phrases for higher confidence
SIGNATURE_PHRASES = {
    "aadhaar_card": [
        ("unique identification authority of india", 4.0),
        ("government of india", 2.0),
        ("mera aadhaar", 4.0),
        ("enrolment no", 3.0),
        ("uidai", 3.0),
        (r"\b\d{4}\s\d{4}\s\d{4}\b", 4.0)  # 12-digit spaced pattern
    ],
    "pan_card": [
        ("income tax department", 4.0),
        ("permanent account number", 4.0),
        ("pan card", 3.0),
        ("govt. of india", 2.0),
        (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 5.0)  # Exact PAN regex pattern
    ],
    "driving_licence": [
        ("driving licence", 4.0),
        ("driving license", 4.0),
        ("union of india", 2.0),
        ("transport department", 3.0),
        ("form 7", 4.0),
        (r"\bdl\s*no\b", 3.0),
        (r"\b[A-Z]{2}[0-9]{2}[ -]?[0-9]{11}\b", 4.0)
    ],
    "passport": [
        ("passport", 3.0),
        ("republic of india", 3.0),
        ("p<ind", 6.0),  # MRZ Line 1 signature
        ("type p", 3.0),
        ("mrz", 3.0)
    ],
    "nach_ecs_mandate": [
        ("nach", 4.0),
        ("ecs", 3.0),
        ("mandate form", 4.0),
        ("sponsor bank code", 3.0),
        ("utility code", 3.0),
        ("umrn", 4.0),
        ("ifsc", 2.5),
        ("bank a/c", 2.5),
        ("debit type", 2.0)
    ],
    "fatca_annexure_form": [
        ("fatca", 5.0),
        ("crs", 4.0),
        ("tax residency", 4.0),
        ("foreign account tax compliance", 5.0),
        ("tin", 2.5),
        ("country of tax residence", 3.5)
    ],
    "benefit_illustration_declaration": [
        ("benefit illustration", 5.0),
        ("understood the features", 4.0),
        ("guaranteed and non-guaranteed", 4.0),
        ("projected investment returns", 4.0),
        ("policyholder declaration", 3.5)
    ],
    "moral_hazard_questionnaire": [
        ("moral hazard", 5.0),
        ("questionnaire", 3.0),
        ("mhq", 5.0),
        ("confidential report", 3.5),
        ("nominee relationship", 3.0),
        ("insurable interest", 3.5)
    ],
    "multiple_policies_consent_form": [
        ("multiple policies", 5.0),
        ("consent form", 4.0),
        ("reason for multiple policies", 5.0),
        ("existing insurance", 3.0),
        ("total sum assured", 3.0)
    ],
    "suitability_profiler_declaration": [
        ("suitability profiler", 5.0),
        ("suitability report", 4.0),
        ("specified person", 4.0),
        ("agent/sp", 3.5),
        ("need analysis", 3.5),
        ("risk profile", 3.0)
    ]
}

class DocumentClassifier:
    def __init__(self):
        self.doc_types = DOCUMENT_TYPES

    def classify_text(self, text: str, filename: Optional[str] = None) -> DocumentTypeInfo:
        """
        Classifies document based on text, regex signals, and optional filename metadata.
        Returns DocumentTypeInfo with predicted category, confidence, and reasons.
        """
        fn_lower = filename.lower() if filename else ""

        if not text or len(text.strip()) < 10:
            # Check filename clues as fallback for image scans / non-searchable PDFs
            if fn_lower:
                for dt, meta in self.doc_types.items():
                    # Check if doc_type key or keywords appear in filename
                    if dt in fn_lower or any(kw.replace(" ", "") in fn_lower.replace("_", "").replace("-", "") for kw in meta.get("keywords", [])[:3]):
                        return DocumentTypeInfo(
                            value=dt,
                            display_name=meta.get("name", dt),
                            confidence=0.75,
                            classification_method="ocr_and_rules",
                            classification_reasons=[f"Identified category from document filename ('{filename}')"]
                        )
                # Specific common tokens
                if "aadhaar" in fn_lower:
                    return DocumentTypeInfo(value="aadhaar_card", display_name="Aadhaar Card", confidence=0.75, classification_method="ocr_and_rules", classification_reasons=["Matched 'aadhaar' in filename"])
                elif "pan" in fn_lower:
                    return DocumentTypeInfo(value="pan_card", display_name="PAN Card", confidence=0.75, classification_method="ocr_and_rules", classification_reasons=["Matched 'pan' in filename"])
                elif "passport" in fn_lower:
                    return DocumentTypeInfo(value="passport", display_name="Passport", confidence=0.75, classification_method="ocr_and_rules", classification_reasons=["Matched 'passport' in filename"])
                elif "driving" in fn_lower or "licence" in fn_lower or "license" in fn_lower:
                    return DocumentTypeInfo(value="driving_licence", display_name="Driving Licence", confidence=0.75, classification_method="ocr_and_rules", classification_reasons=["Matched driving licence keywords in filename"])
                elif "nach" in fn_lower or "ecs" in fn_lower or "mandate" in fn_lower:
                    return DocumentTypeInfo(value="nach_ecs_mandate", display_name="NACH / ECS Mandate", confidence=0.75, classification_method="ocr_and_rules", classification_reasons=["Matched mandate keywords in filename"])
                elif "fatca" in fn_lower:
                    return DocumentTypeInfo(value="fatca_annexure_form", display_name="FATCA Annexure Form", confidence=0.75, classification_method="ocr_and_rules", classification_reasons=["Matched 'fatca' in filename"])
                elif "benefit" in fn_lower or "illustration" in fn_lower:
                    return DocumentTypeInfo(value="benefit_illustration_declaration", display_name="Benefit Illustration Declaration", confidence=0.75, classification_method="ocr_and_rules", classification_reasons=["Matched benefit illustration keywords in filename"])
                elif "moral" in fn_lower or "hazard" in fn_lower or "mhq" in fn_lower:
                    return DocumentTypeInfo(value="moral_hazard_questionnaire", display_name="Moral Hazard Questionnaire", confidence=0.75, classification_method="ocr_and_rules", classification_reasons=["Matched moral hazard keywords in filename"])
                elif "multiple" in fn_lower or "consent" in fn_lower:
                    return DocumentTypeInfo(value="multiple_policies_consent_form", display_name="Multiple Policies Consent Form", confidence=0.75, classification_method="ocr_and_rules", classification_reasons=["Matched consent keywords in filename"])
                elif "suitability" in fn_lower or "profiler" in fn_lower:
                    return DocumentTypeInfo(value="suitability_profiler_declaration", display_name="Suitability Profiler Declaration", confidence=0.75, classification_method="ocr_and_rules", classification_reasons=["Matched suitability keywords in filename"])

            return DocumentTypeInfo(
                value="unsupported_ambiguous",
                display_name="Unsupported / Ambiguous Document",
                confidence=0.0,
                classification_method="ocr_and_rules",
                classification_reasons=["Insufficient text or visual signals extracted for classification"]
            )

        text_lower = text.lower()
        scores: Dict[str, float] = {k: 0.0 for k in self.doc_types.keys()}
        reasons: Dict[str, List[str]] = {k: [] for k in self.doc_types.keys()}

        # Evaluate signature phrases and regexes
        for doc_type, phrases in SIGNATURE_PHRASES.items():
            for item in phrases:
                pattern, weight = item
                if pattern.startswith(r"\b"):
                    # Regex match
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        scores[doc_type] += weight * min(len(matches), 2)
                        reasons[doc_type].append(f"Matched pattern '{pattern}'")
                else:
                    # Substring match
                    if pattern in text_lower:
                        scores[doc_type] += weight
                        reasons[doc_type].append(f"Found keyword '{pattern}'")

        # Find best match
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_type, top_score = sorted_scores[0]
        second_type, second_score = sorted_scores[1] if len(sorted_scores) > 1 else ("", 0.0)

        # Confidence calibration based on score and separation
        if top_score < 4.0:
            # Low signal -> classify as ambiguous/unsupported
            return DocumentTypeInfo(
                value="unsupported_ambiguous",
                display_name="Unsupported / Ambiguous Document",
                confidence=round(min(0.40, top_score / 10.0), 3),
                classification_method="ocr_and_rules",
                classification_reasons=[f"Highest candidate '{top_type}' score ({top_score:.1f}) is below confidence threshold"]
            )

        # Score separation: if second candidate is too close, reduce confidence
        margin = top_score - second_score
        base_conf = min(0.98, 0.65 + (top_score / 20.0))
        if margin < 2.0 and second_score > 3.0:
            base_conf = max(0.50, base_conf - 0.25)
            reasons[top_type].append(f"Close competing candidate: {second_type} (score {second_score:.1f})")

        display_name = self.doc_types.get(top_type, {}).get("name", top_type)

        return DocumentTypeInfo(
            value=top_type,
            display_name=display_name,
            confidence=round(base_conf, 3),
            classification_method="ocr_and_rules",
            classification_reasons=reasons[top_type]
        )

    def manual_override(self, current_info: DocumentTypeInfo, new_type_value: str) -> DocumentTypeInfo:
        """Records a user manual classification correction with audit trail."""
        display_name = self.doc_types.get(new_type_value, {}).get("name", new_type_value)
        return DocumentTypeInfo(
            value=new_type_value,
            display_name=display_name,
            confidence=1.0,
            classification_method="manual_override",
            original_predicted_type=current_info.original_predicted_type or current_info.value,
            classification_reasons=[f"Manually assigned by reviewer (previously '{current_info.value}')"]
        )
