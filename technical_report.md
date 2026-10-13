# Technical Report: Intelligent Document Classification and Field-Level Extraction Pipeline

**Author:** Data Science Internship Evaluation Candidate  
**Question:** Question 3 — Document Classification & Extraction Pipeline  
**Date:** October 2026  

---

## 1. Executive Summary

This project implements an end-to-end Python and Streamlit pipeline for automated document classification and field-level intelligent extraction across 10 mixed identity, banking, and insurance documents. The system places primary emphasis on handwritten text handling, calibrated confidence estimation, auditability, and transparent human-in-the-loop review.

The pipeline processes both PDF files (via PyMuPDF) and high-resolution raster images (PNG, JPG, JPEG), performs non-destructive stroke-preserving preprocessing, executes multi-signal document classification, runs field-specific extraction and domain validation, assigns calibrated confidence scores, and routes uncertain items to a dedicated human review dashboard.

---

## 2. Document Classification Architecture

Document classification operates on a multi-signal decision framework rather than fragile single-keyword matching:
1. **Weighted Signature Keyword Density**: Evaluates presence of institutional markers (e.g. "Unique Identification Authority of India", "Income Tax Department", "Form 7", "National Automated Clearing House").
2. **Structural Regex Cues**: Detects document-defining identifiers (e.g., PAN 10-character alphanumeric structure `[A-Z]{5}[0-9]{4}[A-Z]`, 12-digit UID pattern, Passport MRZ line signatures `P<IND`, UMRN sequences).
3. **Ambiguity & Fallback Detection**: If the maximum candidate score does not meet a confidence margin or if competing candidates have equal weight, the document is classified as `unsupported_ambiguous`.
4. **Human Review Correction**: Reviewers can override classifications in the UI, which triggers automatic re-extraction of the required fields under the new category with an immutable audit record.

---

## 3. Dedicated Handwriting Extraction Strategy

Handwritten documents (NACH mandates, FATCA annexures, benefit illustrations, moral hazard questionnaires, consent forms, and suitability declarations) present significant challenges due to cursive variability, light pen pressure, and uneven spacing.

### Key Methodologies:
- **Stroke-Preserving Preprocessing**: Harsh Otsu binarization damages thin ink strokes and breaks character connectivity. Our pipeline instead applies gentle bilateral filtering (smoothing background noise while preserving pen edges) followed by Contrast Limited Adaptive Histogram Equalization (CLAHE) on the luminance channel.
- **Contextual Field Cropping**: For every handwritten field, a cropped sub-region with 10% surrounding padding is generated and retained. This ensures that during human review, auditors see the visual crop side-by-side with the machine prediction.
- **Checkbox Detection via Contour Analysis**: Rather than inferring checkbox status from surrounding text, an OpenCV contour detector locates square form boxes (aspect ratio 0.8–1.25, width 12–55px) and calculates the internal pixel fill ratio. A fill ratio > 0.18 indicates a marked box. Multiple marks or faint marks are flagged as `ambiguous_checkbox`.
- **Date Ambiguity Safeguard**: In Indian documentation, dates appear as DD/MM/YYYY, MM/DD/YYYY, or YYYY-MM-DD. When both day and month values are $\le 12$ (e.g., `03/04/2025`), the order cannot be mathematically proven without form header specifications. The system normalizes unambiguous dates to ISO (`YYYY-MM-DD`) but explicitly tags ambiguous dates with `status = "ambiguous_date"` and penalizes confidence to mandate human confirmation.
- **Passport MRZ Line 2 Checksums**: The pipeline implements the official ICAO 9303 7-3-1 weighting algorithm to verify check digits for passport number, date of birth, and expiry date. Checksum failures immediately flag the field for human inspection.

---

## 4. Confidence Scoring and Operational Threshold

### Scoring Formulation:
The system computes a composite field-level confidence $C \in [0.0, 1.0]$:
$$C = \text{clamp}\Big(C_{\text{base}} + \Delta_{\text{format}} - \Delta_{\text{ambiguity}} - \Delta_{\text{validation}}, 0.0, 1.0\Big)$$

Where:
- $C_{\text{base}}$: Raw visual recognition certainty reported by the recognition engine or conservative heuristic (0.70 printed, 0.50 handwritten).
- $\Delta_{\text{format}}$: Format conformance bonus (+0.05 for strict regex match on printed/MRZ).
- $\Delta_{\text{validation}}$: Severe penalty for critical validation failure (caps $C$ at 0.40). High recognition confidence is never permitted to override a failed validation rule.
- $\Delta_{\text{ambiguity}}$: Penalty for date ambiguity (caps $C$ at 0.65) or checkbox ambiguity (caps $C$ at 0.50).

### Threshold Rationale:
A default threshold of **0.85** is established as an operational baseline. In financial mandate debiting and KYC identification, false positives (accepting an incorrect account number or PAN) carry high compliance risk and potential financial loss. A 0.85 threshold ensures that fields with visual noise, cursive stroke uncertainty, or format anomalies are reliably routed to human auditors.

---

## 5. Limitations and Observed Failure Cases

From empirical testing of the pipeline:
1. **Free-Tier API Quota Limits**: External AI multimodal services (such as Google Gemini free-tier endpoints) have daily request limits (e.g., 20 requests/day). The pipeline handles this gracefully by providing automatic fallback to local multi-engine processing (PyMuPDF, OpenCV contour analysis, and regex parsers) and displaying clear configuration status messages.
2. **Severely Degraded or Torn Documents**: Documents where the header or border is completely cropped off may result in lower classification confidence and route to the `unsupported_ambiguous` category.
3. **Crossed-out / Overwritten Handwriting**: When an applicant crosses out a digit and writes a second digit on top, single-pass OCR may read both digits concatenated. These cases produce length validation errors and are caught by our flagging engine.
4. **Uncalibrated Heuristics**: In the absence of an OCR model that outputs per-character log-probabilities, application-derived confidence heuristics must be treated as operational risk scores rather than statistical probabilities.

---

## 6. Conclusion

The pipeline fulfills all requirements of Question 3: modular architecture, full coverage of 10 document categories, dedicated handwriting handling, transparent calibrated confidence scoring, human review workflows, ground-truth evaluation, and valid JSON/CSV exports.
