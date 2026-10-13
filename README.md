# Intelligent Document Classification and Field-Level Extraction Pipeline

**Question 3 — Data Science Internship Evaluation**  
AI-powered Document Processing Pipeline that classifies 10 mixed identity and insurance documents, extracts required fields, assigns calibrated field-level confidence scores, routes uncertain fields to an interactive human-review dashboard, and evaluates performance against ground-truth answer keys.

---

## 1. Supported Document Types and Required Fields

The pipeline implements dedicated extraction configurations for all 10 document categories:

| Document Category | Document Type Key | Category Type | Primary Required Fields |
|---|---|---|---|
| **A. Aadhaar Card** | `aadhaar_card` | Identity (Printed) | Aadhaar Number, Full Name, Date of Birth, Address |
| **B. PAN Card** | `pan_card` | Identity (Printed) | PAN Number, Full Name, Father's Name, Date of Birth |
| **C. Driving Licence** | `driving_licence` | Identity (Printed) | DL Number, Name, Date of Issue, Valid Till Date |
| **D. Passport** | `passport` | Identity / MRZ | Passport Number, Date of Birth, Date of Expiry, MRZ Line 2 |
| **E. NACH / ECS Mandate** | `nach_ecs_mandate` | Banking / Handwritten | Bank Account Number, IFSC Code, Bank Name, Amount in Figures, Frequency |
| **F. FATCA Annexure Form** | `fatca_annexure_form` | Insurance / Handwritten | Policy Number, TIN / PAN, Father's Name, Place of Birth, Nationality |
| **G. Benefit Illustration Declaration** | `benefit_illustration_declaration` | Insurance / Handwritten | Application Number, Policyholder Name, Date, Place |
| **H. Moral Hazard Questionnaire** | `moral_hazard_questionnaire` | Insurance / Handwritten | Application Number, Name of Life Assured, Nominee Relationship, Date, Place |
| **I. Multiple Policies Consent Form** | `multiple_policies_consent_form` | Insurance / Checkbox | Proposer Name, Reason for Multiple Policies (Selected Checkbox), Date, Place |
| **J. Suitability Profiler Declaration** | `suitability_profiler_declaration` | Insurance / Handwritten | Application Number, Name of Life Assured, Name of Agent/SP, Date, Place |

---

## 2. Architecture and Project Structure

```
document_extraction_pipeline/
│
├── app.py                     # Streamlit multi-tab web application
├── requirements.txt           # Python dependency requirements
├── README.md                  # System documentation & execution guide
├── config.py                  # Paths, environment configs, threshold settings
├── .env                       # Environment variables and API keys
│
├── configs/
│   └── document_types.json    # JSON specifications for all 10 document types
│
├── schemas/
│   ├── document_schema.py     # DocumentRecord and DocumentTypeInfo Pydantic schemas
│   └── field_schema.py        # FieldExtractionResult schema with audit trail & masking
│
├── services/
│   ├── document_loader.py     # Batch ingestion & PyMuPDF PDF page renderer
│   ├── preprocessing.py       # Deskew, CLAHE contrast enhancement, safe crop generator
│   ├── document_classifier.py # Multi-signal weighted keyword & regex classifier
│   ├── ocr_engine.py          # Unified OCR engine (PyMuPDF, Tesseract, Gemini Vision)
│   ├── handwriting_engine.py  # Handwriting crop generation & OpenCV checkbox analyzer
│   ├── field_extractor.py     # Coordinates parsing, regex rules & domain validators
│   ├── validators.py          # Verhoeff (Aadhaar), ICAO 9303 (MRZ), IFSC, Dates
│   ├── confidence.py          # Calibrated confidence scoring & threshold routing
│   ├── review_service.py      # Human-in-the-loop review, audits & confirmation
│   ├── evaluation.py          # Ground-truth evaluation metrics & breakdown tables
│   ├── export_service.py      # Structured JSON & CSV report generators
│   └── sample_generator.py    # Generates 10 synthetic evaluation docs & ground truth
│
├── sample_documents/          # 10 synthetic evaluation documents (PDFs & PNGs)
│   ├── ground_truth_answer_key.json
│   └── ground_truth_answer_key.csv
│
└── tests/                     # Comprehensive Pytest test suite (21 unit tests)
    ├── test_validators.py
    ├── test_classification.py
    ├── test_confidence.py
    ├── test_json_export.py
    ├── test_evaluation.py
    └── test_pipeline_e2e.py
```

---

## 3. Dedicated Handwriting Extraction Approach

Handwritten forms are the primary focus of this evaluation:

1. **Stroke-Preserving Preprocessing**: Standard harsh binarization destroys thin ballpoint pen strokes. The pipeline applies a bilateral filter followed by Contrast Limited Adaptive Histogram Equalization (CLAHE) on grayscale images to enhance pen ink contrast while avoiding stroke erosion.
2. **Contextual Crop Retention**: For any extracted handwritten field, an enlarged contextual bounding crop is saved and displayed in the review interface so human auditors can inspect the actual visual evidence.
3. **Format Validation Is Not Evidence**: A syntactically valid IFSC or account number does not guarantee visual correctness. The system scores raw visual recognition confidence separately from format validation.
4. **Checkbox Recognition**: An OpenCV contour analyzer inspects checkbox bounding boxes and internal pixel density (fill ratio > 0.18) to associate marked checkboxes with form reasons and detect ambiguous multi-markings.
5. **Date Ambiguity Handling**: Dates where day and month are both $\le 12$ (e.g., `03/04/2025`) cannot be definitively resolved without explicit form header guidelines. The system tags them as `ambiguous_date` and routes them to human review.

---

## 4. Confidence Threshold Rationale

- **Default Threshold**: `0.85`
- **Operational Rationale**: In identity verification (KYC), banking mandates (NACH/ECS), and insurance underwriting (declarations), incorrect extractions cause financial fraud and regulatory penalties. A baseline of `0.85` ensures that any degraded ink strokes, unverified check digits, or ambiguous date orders are routed to human review.
- **Configurability**: The threshold is fully configurable in both the UI sidebar and the settings tab (from 0.50 to 0.99).

---

## 5. Execution Instructions (Windows PowerShell)

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Run Unit Tests
```powershell
python -m pytest tests/
```
All 21 unit tests will run and pass.

### Step 3: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 6. Testing the Pipeline with 10 Documents

1. Open the application.
2. Navigate to **2. Upload Documents**.
3. Click the **"📦 Load 10 Evaluation Documents"** button. This automatically loads all 10 evaluation documents (PDFs and PNGs) across all 10 categories.
4. View real telemetry on the **1. Overview Dashboard**.
5. Inspect classifications on **3. Document Classification** and test manual overrides.
6. Verify extracted fields on **4. Field Extraction Results**.
7. Audit handwritten fields and edit values on **5. Handwriting Review**.
8. View flagged fields and approve them on **6. Flagged Fields**.
9. Review precision, recall, and exact-match rates against ground truth on **7. Ground-Truth Evaluation**.
10. Download single document JSONs, batch JSON, or CSVs on **8. JSON & CSV Exports**.

---

## 7. Privacy and Security Disclosures

- **Local Processing**: Ingestion, PDF rendering, deskewing, and regex extractions execute locally.
- **Sensitive Data Masking**: Aadhaar numbers (`XXXX-XXXX-3821`), PAN cards (`ABCPS****F`), and Bank Account numbers (`******1012`) are masked in UI tables and exports by default.
- **No Retraining**: External APIs (when enabled) are called strictly for inference over TLS.
