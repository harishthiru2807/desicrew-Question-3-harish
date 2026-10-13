"""
Intelligent Document Classification and Field-Level Extraction Pipeline
Streamlit Web Application
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
from PIL import Image

# Pipeline imports
from config import (
    DOCUMENT_TYPES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    ENABLE_SENSITIVE_DATA_MASKING,
    SAMPLE_DOCS_DIR,
    EXPORTS_DIR,
    get_document_type_names
)
from schemas.document_schema import DocumentRecord, DocumentTypeInfo
from schemas.field_schema import FieldExtractionResult
from services.document_loader import DocumentLoader
from services.document_classifier import DocumentClassifier
from services.field_extractor import FieldExtractor
from services.ocr_engine import OCREngine
from services.gemini_vision import GeminiVisionClient
from services.review_service import ReviewService
from services.evaluation import EvaluationService
from services.export_service import ExportService
from services.sample_generator import create_synthetic_documents

# Streamlit Page Config
st.set_page_config(
    page_title="DocSense AI - Document Extraction Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light Theme CSS
st.markdown("""
<style>
    /* Clean Light Theme Styling */
    .main {
        background-color: #F8FAFC;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .badge-valid {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-flagged {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-handwritten {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-mrz {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 6px;
        padding: 8px 16px;
        color: #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
if "mask_sensitive_data" not in st.session_state:
    st.session_state.mask_sensitive_data = ENABLE_SENSITIVE_DATA_MASKING
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if "active_reviewer" not in st.session_state:
    st.session_state.active_reviewer = "Auditor_01"

# Initialize Services
gemini_client = GeminiVisionClient(api_key=st.session_state.gemini_api_key)
ocr_engine = OCREngine(gemini_client=gemini_client)
doc_loader = DocumentLoader()
classifier = DocumentClassifier()
field_extractor = FieldExtractor(confidence_threshold=st.session_state.confidence_threshold)

# SIDEBAR CONTROLS
with st.sidebar:
    st.title("📄 DocSense AI")
    st.caption("Intelligent Document & Handwriting Pipeline")
    st.divider()

    st.subheader("⚙️ Pipeline Controls")
    new_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.99,
        value=float(st.session_state.confidence_threshold),
        step=0.01,
        help="Default 0.85 operational baseline. Fields with confidence below this threshold are routed to human review."
    )
    if new_threshold != st.session_state.confidence_threshold:
        st.session_state.confidence_threshold = new_threshold
        field_extractor.set_threshold(new_threshold)

    st.session_state.mask_sensitive_data = st.checkbox(
        "Mask Sensitive Data (KYC / Banking)",
        value=st.session_state.mask_sensitive_data,
        help="Masks Aadhaar, PAN, and Bank Account numbers in UI summaries for data privacy."
    )

    st.session_state.active_reviewer = st.text_input(
        "Reviewer Name / ID",
        value=st.session_state.active_reviewer
    )

    st.divider()

    # Engine Status Display
    status_info = ocr_engine.get_engine_status()
    st.markdown("**Engine Status:**")
    st.markdown("🟢 **PyMuPDF**: Ready")
    
    if status_info["gemini_vision"]["available"]:
        st.markdown(f"🟢 **Gemini Vision**: Ready")
    else:
        st.markdown(f"🟡 **Gemini Vision**: Local Fallback")
        st.caption(f"{status_info['gemini_vision']['message']}")

    if status_info["tesseract"]["available"]:
        st.markdown("🟢 **Tesseract OCR**: Ready")
    else:
        st.markdown("⚪ **Tesseract OCR**: Not in PATH (using local engine)")

    st.divider()

    # Session Management
    if st.button("🗑️ Clear Session Data", use_container_width=True):
        doc_loader.clear_session_data()
        st.session_state.processed_documents = []
        st.success("Session data cleared successfully.")
        st.rerun()

# Processing Function
def process_batch(uploaded_files, is_demo: bool = False):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    docs_to_add = []
    total_files = len(uploaded_files)

    for i, file_obj in enumerate(uploaded_files):
        filename = file_obj.name if hasattr(file_obj, "name") else file_obj
        status_text.text(f"Processing ({i+1}/{total_files}): {filename}...")
        
        # Read bytes
        if hasattr(file_obj, "read"):
            bytes_data = file_obj.read()
        else:
            with open(file_obj, "rb") as f:
                bytes_data = f.read()

        t0 = time.time()
        # 1. Ingest & Preprocess
        load_res = doc_loader.load_document(bytes_data, filename, doc_index=len(st.session_state.processed_documents) + len(docs_to_add) + 1)
        
        if load_res["status"] not in ["loaded", "pending"]:
            # Record corrupt / unsupported
            err_doc = DocumentRecord(
                document_id=load_res["document_id"],
                filename=filename,
                file_path=load_res.get("file_path", ""),
                document_type=DocumentTypeInfo(
                    value="unsupported_ambiguous",
                    display_name="Unsupported / Corrupt File",
                    confidence=0.0,
                    classification_method="ocr_and_rules",
                    classification_reasons=[load_res.get("error", "Error reading file")]
                ),
                processing_error=load_res.get("error"),
                is_ambiguous_or_unsupported=True
            )
            docs_to_add.append(err_doc)
            continue

        # 2. Text and Vision Extraction
        extracted_text = load_res.get("digital_text", "")
        img_path = load_res["image_paths"][0] if load_res["image_paths"] else None

        # Try Gemini Vision if available
        vision_result = None
        if gemini_client.is_available()[0] and img_path:
            v_ok, v_data, _ = gemini_client.analyze_document(img_path)
            if v_ok:
                vision_result = v_data
                if vision_result.get("full_text"):
                    extracted_text = (extracted_text + "\n" + vision_result["full_text"]).strip()

        # 3. Document Classification (using text, regexes, and filename metadata)
        class_info = classifier.classify_text(extracted_text, filename=filename)

        # 4. Field Extraction
        crop_dir = Path(load_res["file_path"]).parent if load_res.get("file_path") else None
        extracted_fields = field_extractor.extract_document_fields(
            doc_type=class_info.value,
            text=extracted_text,
            vision_result=vision_result,
            image_path=img_path,
            crop_dir=crop_dir
        )

        t_elapsed = time.time() - t0

        doc_record = DocumentRecord(
            document_id=load_res["document_id"],
            filename=filename,
            file_path=load_res.get("file_path", ""),
            page_count=load_res.get("page_count", 1),
            image_paths=load_res.get("image_paths", []),
            original_image_paths=load_res.get("original_image_paths", []),
            document_type=class_info,
            fields=extracted_fields,
            processing_time_seconds=t_elapsed,
            is_ambiguous_or_unsupported=(class_info.value == "unsupported_ambiguous")
        )
        docs_to_add.append(doc_record)
        progress_bar.progress((i + 1) / total_files)

    st.session_state.processed_documents.extend(docs_to_add)
    progress_bar.empty()
    status_text.empty()
    st.success(f"Successfully processed {len(docs_to_add)} documents.")
    st.rerun()

# MAIN NAVIGATION TABS
tabs = st.tabs([
    "1. 📊 Overview Dashboard",
    "2. 📥 Upload Documents",
    "3. 🏷️ Document Classification",
    "4. 🔍 Field Extraction Results",
    "5. ✍️ Handwriting Review",
    "6. 🚩 Flagged Fields",
    "7. 🎯 Ground-Truth Evaluation",
    "8. 💾 JSON & CSV Exports",
    "9. ⚙️ Configuration"
])

# ==============================================================================
# TAB 1: OVERVIEW DASHBOARD
# ==============================================================================
with tabs[0]:
    st.header("Overview Dashboard")
    st.markdown("Real-time telemetry and extraction statistics calculated across all active documents.")

    docs = st.session_state.processed_documents
    total_docs = len(docs)
    
    # Calculate real statistics
    if total_docs > 0:
        _, metrics = ExportService.generate_flagging_report(
            docs,
            confidence_threshold=st.session_state.confidence_threshold,
            mask_sensitive=st.session_state.mask_sensitive_data
        )
    else:
        metrics = {
            "total_documents": 0,
            "classified_successfully": 0,
            "classification_failures": 0,
            "total_fields_expected": 0,
            "total_fields_extracted": 0,
            "total_fields_flagged": 0,
            "missing_fields": 0,
            "invalid_fields": 0,
            "handwritten_fields_flagged": 0,
            "extraction_rate_pct": 0.0,
            "flagging_rate_pct": 0.0
        }

    # Metric Cards Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Documents</div>
            <div class="metric-value">{metrics['total_documents']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Classified Successfully</div>
            <div class="metric-value">{metrics['classified_successfully']} / {metrics['total_documents']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Extraction Rate</div>
            <div class="metric-value">{metrics['extraction_rate_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Flagged for Review</div>
            <div class="metric-value" style="color: #DC2626;">{metrics['total_fields_flagged']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Metric Cards Row 2
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Handwritten Flagged</div>
            <div class="metric-value">{metrics['handwritten_fields_flagged']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Missing Fields</div>
            <div class="metric-value">{metrics['missing_fields']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c7:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Invalid Fields</div>
            <div class="metric-value">{metrics['invalid_fields']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c8:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Operational Threshold</div>
            <div class="metric-value">{st.session_state.confidence_threshold:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    if total_docs == 0:
        st.info("💡 No documents loaded yet. Go to **Upload Documents** or click **'Load 10 Evaluation Documents'** to run the complete pipeline.")
    else:
        st.subheader("Document Ingestion Summary")
        summary_rows = []
        for d in docs:
            flagged = d.get_flagged_fields_count()
            hw_flagged = d.get_handwritten_flagged_count()
            summary_rows.append({
                "Document ID": d.document_id,
                "Filename": d.filename,
                "Predicted Type": d.document_type.display_name,
                "Classification Confidence": f"{d.document_type.confidence:.2f}",
                "Classification Method": d.document_type.classification_method,
                "Total Fields": len(d.fields),
                "Flagged Fields": flagged,
                "Handwritten Flagged": hw_flagged,
                "Review Status": d.document_review_status.upper()
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# ==============================================================================
# TAB 2: UPLOAD DOCUMENTS
# ==============================================================================
with tabs[1]:
    st.header("Upload Documents")
    st.markdown("Accepts batch uploads of PDF, PNG, JPG, and JPEG documents (up to 10+ documents).")

    col_up, col_demo = st.columns([2, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "Choose documents for batch processing",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("🚀 Process Uploaded Batch", type="primary"):
                process_batch(uploaded_files)

    with col_demo:
        st.markdown("#### ⚡ Evaluation Demo Set")
        st.caption("Load 10 evaluation documents covering all 10 identity and handwritten insurance forms.")
        if st.button("📦 Load 10 Evaluation Documents", use_container_width=True):
            # Ensure sample documents exist
            if not any(SAMPLE_DOCS_DIR.glob("DOC_*.*")):
                create_synthetic_documents()
            sample_files = sorted(list(SAMPLE_DOCS_DIR.glob("DOC_*.*")))
            if sample_files:
                process_batch(sample_files, is_demo=True)
            else:
                st.error("Sample documents could not be generated.")

    st.divider()
    st.subheader("Supported Document Categories (All 10 Types)")
    doc_cols = st.columns(2)
    with doc_cols[0]:
        st.markdown("""
        **Identity Documents (Printed & MRZ):**
        - **Aadhaar Card**: Aadhaar Number, Full Name, DOB, Address
        - **PAN Card**: PAN Number, Full Name, Father's Name, DOB
        - **Driving Licence**: DL Number, Name, Issue Date, Valid Till
        - **Passport**: Passport Number, DOB, Expiry Date, MRZ Line 2
        """)
    with doc_cols[1]:
        st.markdown("""
        **Banking & Insurance Forms (Handwritten Focus):**
        - **NACH / ECS Mandate**: Account Number, IFSC, Bank, Amount, Frequency
        - **FATCA Annexure**: Policy No, TIN/PAN, Father's Name, Birth Place, Nationality
        - **Benefit Illustration**: Application No, Policyholder, Date, Place
        - **Moral Hazard Questionnaire**: Application No, Life Assured, Nominee, Date, Place
        - **Multiple Policies Consent**: Proposer Name, Reason Checkbox, Date, Place
        - **Suitability Profiler**: Application No, Life Assured, Agent/SP, Date, Place
        """)

# ==============================================================================
# TAB 3: DOCUMENT CLASSIFICATION
# ==============================================================================
with tabs[2]:
    st.header("Document Classification")
    st.markdown("Multi-signal classifier using OCR text, keyword density, structural regexes, and layout markers.")

    docs = st.session_state.processed_documents
    if not docs:
        st.info("No documents loaded. Please upload documents first.")
    else:
        for doc in docs:
            with st.expander(f"📄 {doc.document_id} — {doc.filename} [{doc.document_type.display_name}]", expanded=False):
                col_img, col_info = st.columns([1, 2])
                with col_img:
                    if doc.original_image_paths and os.path.exists(doc.original_image_paths[0]):
                        st.image(doc.original_image_paths[0], caption="Document Page 1", use_container_width=True)
                with col_info:
                    st.markdown(f"**Predicted Document Type:** `{doc.document_type.display_name}`")
                    st.markdown(f"**Classification Confidence:** `{doc.document_type.confidence:.2f}`")
                    st.markdown(f"**Method:** `{doc.document_type.classification_method}`")
                    
                    if doc.document_type.classification_reasons:
                        st.markdown("**Evidence Signals:**")
                        for r in doc.document_type.classification_reasons:
                            st.markdown(f"- {r}")

                    st.markdown("---")
                    st.markdown("#### ✏️ Manual Classification Correction")
                    type_options = list(DOCUMENT_TYPES.keys())
                    current_idx = type_options.index(doc.document_type.value) if doc.document_type.value in type_options else 0
                    
                    new_selected_type = st.selectbox(
                        "Correct Document Type",
                        options=type_options,
                        format_func=lambda k: DOCUMENT_TYPES.get(k, {}).get("name", k),
                        index=current_idx,
                        key=f"correct_type_{doc.document_id}"
                    )

                    if st.button(f"Save Correction for {doc.document_id}", key=f"btn_corr_{doc.document_id}"):
                        if new_selected_type != doc.document_type.value:
                            # Update type
                            doc.document_type = classifier.manual_override(doc.document_type, new_selected_type)
                            # Re-run field extraction for corrected type
                            extracted_text = ""
                            img_p = doc.image_paths[0] if doc.image_paths else None
                            doc.fields = field_extractor.extract_document_fields(
                                doc_type=new_selected_type,
                                text=extracted_text,
                                image_path=img_p
                            )
                            st.success(f"Updated {doc.document_id} to '{DOCUMENT_TYPES[new_selected_type]['name']}' and re-extracted fields.")
                            st.rerun()

# ==============================================================================
# TAB 4: FIELD EXTRACTION RESULTS
# ==============================================================================
with tabs[3]:
    st.header("Field Extraction Results")
    docs = st.session_state.processed_documents
    if not docs:
        st.info("No documents loaded. Please upload documents first.")
    else:
        df_all = ExportService.generate_field_extraction_dataframe(
            docs,
            mask_sensitive=st.session_state.mask_sensitive_data
        )

        if df_all.empty:
            st.warning("⚠️ No fields could be extracted from the uploaded document(s). The document may be an image scan without embedded digital text, or classified as unsupported/ambiguous. You can assign the document type in the **3. Document Classification** tab to define its expected fields.")
        else:
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                search_query = st.text_input("🔍 Search Fields or Values", "")
            with col_f2:
                filter_doc = st.selectbox(
                    "Filter by Document",
                    options=["All Documents"] + [f"{d.document_id} ({d.filename})" for d in docs]
                )

            filtered_df = df_all.copy()
            if filter_doc != "All Documents" and "Document ID" in filtered_df.columns:
                doc_id_sel = filter_doc.split(" ")[0]
                filtered_df = filtered_df[filtered_df["Document ID"] == doc_id_sel]

            if search_query and "Field Name" in filtered_df.columns and "Extracted Value" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["Field Name"].astype(str).str.contains(search_query, case=False, na=False) |
                    filtered_df["Extracted Value"].astype(str).str.contains(search_query, case=False, na=False)
                ]

            display_cols = [
                c for c in [
                    "Document ID", "Filename", "Document Type", "Field Name",
                    "Extracted Value", "Confidence", "Source Type", "Validation Status",
                    "Needs Review", "Review Reason", "Review Status"
                ] if c in filtered_df.columns
            ]

            st.dataframe(filtered_df[display_cols], use_container_width=True)

# ==============================================================================
# TAB 5: HANDWRITING REVIEW
# ==============================================================================
with tabs[4]:
    st.header("Handwriting Review Interface")
    st.markdown("Dedicated interface prioritizing handwritten banking mandates, declarations, and questionnaires.")

    docs = st.session_state.processed_documents
    if not docs:
        st.info("No documents loaded. Please upload documents first.")
    else:
        # Filter docs that have handwritten fields
        hw_docs = [d for d in docs if any(f.source_type == "handwritten" for f in d.fields.values())]
        
        if not hw_docs:
            st.info("No handwritten documents found in current batch.")
        else:
            selected_hw_doc_id = st.selectbox(
                "Select Handwritten Form for Inspection",
                options=[f"{d.document_id} — {d.filename} ({d.document_type.display_name})" for d in hw_docs]
            )
            doc_id = selected_hw_doc_id.split(" ")[0]
            current_doc = next(d for d in hw_docs if d.document_id == doc_id)

            c_prev, c_fields = st.columns([1, 1])
            with c_prev:
                st.subheader("Document Visual Preview")
                if current_doc.original_image_paths and os.path.exists(current_doc.original_image_paths[0]):
                    st.image(current_doc.original_image_paths[0], caption="Original Form", use_container_width=True)

            with c_fields:
                st.subheader("Handwritten Fields Audit")
                for fname, f in current_doc.fields.items():
                    if f.source_type == "handwritten":
                        with st.container():
                            st.markdown(f"**{f.display_name or fname}**")
                            col_val, col_meta = st.columns([2, 1])
                            with col_val:
                                current_val = f.reviewed_value if f.review_status == "edited" and f.reviewed_value is not None else (f.value or "")
                                edit_val = st.text_input(
                                    f"Value for {fname}",
                                    value=current_val,
                                    key=f"hw_edit_{doc_id}_{fname}"
                                )
                            with col_meta:
                                conf_color = "#166534" if f.confidence >= st.session_state.confidence_threshold else "#991B1B"
                                st.markdown(f"Confidence: <b style='color:{conf_color};'>{f.confidence:.2f}</b>", unsafe_allow_html=True)
                                st.markdown(f"Status: `{f.validation_status}`")

                            btn_c1, btn_c2, btn_c3 = st.columns(3)
                            with btn_c1:
                                if st.button("✅ Confirm", key=f"btn_hw_conf_{doc_id}_{fname}"):
                                    ReviewService.confirm_field(current_doc, fname, reviewer_id=st.session_state.active_reviewer)
                                    st.success(f"Confirmed {fname}")
                                    st.rerun()
                            with btn_c2:
                                if st.button("💾 Save Edit", key=f"btn_hw_save_{doc_id}_{fname}"):
                                    ReviewService.edit_field_value(current_doc, fname, edit_val, reviewer_id=st.session_state.active_reviewer)
                                    st.success(f"Updated {fname}")
                                    st.rerun()
                            with btn_c3:
                                if st.button("🚫 Unreadable", key=f"btn_hw_unr_{doc_id}_{fname}"):
                                    ReviewService.mark_field_unreadable(current_doc, fname, reviewer_id=st.session_state.active_reviewer)
                                    st.warning(f"Marked {fname} as unreadable")
                                    st.rerun()
                            st.divider()

# ==============================================================================
# TAB 6: FLAGGED FIELDS
# ==============================================================================
with tabs[5]:
    st.header("Flagged Fields Report")
    st.markdown(f"""
    **Operational Threshold:** `{st.session_state.confidence_threshold:.2f}`  
    *Fields are flagged when confidence falls below the threshold, validation fails, or handwriting stroke clarity is ambiguous.*
    """)

    docs = st.session_state.processed_documents
    if not docs:
        st.info("No documents loaded. Please upload documents first.")
    else:
        df_flagged, metrics = ExportService.generate_flagging_report(
            docs,
            confidence_threshold=st.session_state.confidence_threshold,
            mask_sensitive=st.session_state.mask_sensitive_data
        )

        st.markdown(f"**Total Flagged Fields:** `{metrics['total_fields_flagged']}` out of `{metrics['total_fields_expected']}` expected ({metrics['flagging_rate_pct']}%)")
        
        if df_flagged.empty:
            st.success("🎉 No fields flagged! All extracted fields passed validation and met confidence thresholds.")
        else:
            st.dataframe(df_flagged, use_container_width=True)

            st.divider()
            st.subheader("⚡ Bulk Approval")
            if st.button("Approve Remaining Flagged Fields", type="primary"):
                total_confirmed = 0
                for d in docs:
                    total_confirmed += ReviewService.confirm_all_fields(d, reviewer_id=st.session_state.active_reviewer)
                st.success(f"Confirmed {total_confirmed} pending fields across batch.")
                st.rerun()

# ==============================================================================
# TAB 7: GROUND-TRUTH EVALUATION
# ==============================================================================
with tabs[6]:
    st.header("Ground-Truth Evaluation")
    st.markdown("Evaluate model extractions against ground-truth answer keys without fabricating accuracy.")

    docs = st.session_state.processed_documents
    gt_file = st.file_uploader("Upload Ground-Truth File (JSON or CSV)", type=["json", "csv"])

    col_gt_actions = st.columns([2, 1])
    with col_gt_actions[1]:
        # Download template
        sample_gt_path = SAMPLE_DOCS_DIR / "ground_truth_answer_key.json"
        if sample_gt_path.exists():
            with open(sample_gt_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "📥 Download Ground-Truth Key (10 Docs)",
                    data=f.read(),
                    file_name="ground_truth_answer_key.json",
                    mime="application/json",
                    use_container_width=True
                )

    ground_truth_data = None
    if gt_file:
        content = gt_file.read().decode("utf-8")
        ground_truth_data = EvaluationService.parse_ground_truth_file(content, gt_file.name)
    elif (SAMPLE_DOCS_DIR / "ground_truth_answer_key.json").exists() and docs:
        with open(SAMPLE_DOCS_DIR / "ground_truth_answer_key.json", "r", encoding="utf-8") as f:
            content = f.read()
            ground_truth_data = EvaluationService.parse_ground_truth_file(content, "ground_truth_answer_key.json")

    if not docs:
        st.info("No processed documents to evaluate. Ingest documents first.")
    elif not ground_truth_data:
        st.info("Upload a ground-truth JSON/CSV or load the 10-document evaluation demo set to view metrics.")
    else:
        eval_results = EvaluationService.evaluate_batch(docs, ground_truth_data)
        
        if "error" in eval_results:
            st.error(eval_results["error"])
        else:
            st.subheader("📈 Evaluated Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Doc Classification Accuracy", f"{eval_results['classification_accuracy_pct']}%")
            with m2:
                st.metric("Overall Field Exact-Match", f"{eval_results['overall_field_accuracy_pct']}%")
            with m3:
                st.metric("Printed Fields Accuracy", f"{eval_results['printed_accuracy_pct']}%")
            with m4:
                st.metric("Handwritten Fields Accuracy", f"{eval_results['handwritten_accuracy_pct']}%")

            m5, m6, m7, m8 = st.columns(4)
            with m5:
                st.metric("Missing Field Rate", f"{eval_results['missing_field_rate_pct']}%")
            with m6:
                st.metric("False Acceptance Rate", f"{eval_results['false_acceptance_rate_pct']}%")
            with m7:
                st.metric("Review Precision", f"{eval_results['review_precision_pct']}%")
            with m8:
                st.metric("Review Recall", f"{eval_results['review_recall_pct']}%")

            st.divider()
            st.subheader("Field-by-Field Breakdown")
            st.dataframe(pd.DataFrame(eval_results["field_accuracy_breakdown"]), use_container_width=True)

            st.subheader("High-Risk Identifier Subgroups (IFSC, PAN, Dates, etc.)")
            st.dataframe(pd.DataFrame(eval_results["high_risk_breakdown"]), use_container_width=True)

            with st.expander("Detailed Comparison Audit (Prediction vs Ground Truth)"):
                st.dataframe(eval_results["comparison_df"], use_container_width=True)

# ==============================================================================
# TAB 8: JSON & CSV EXPORTS
# ==============================================================================
with tabs[7]:
    st.header("JSON & CSV Exports")
    st.markdown("Export structured, machine-readable data conforming to Section 11 specifications.")

    docs = st.session_state.processed_documents
    if not docs:
        st.info("No documents to export. Please process documents first.")
    else:
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.subheader("📦 Batch Structured JSON")
            batch_json = ExportService.export_batch_json(docs)
            st.download_button(
                "📥 Download Complete Batch JSON",
                data=batch_json,
                file_name="extracted_documents_batch.json",
                mime="application/json",
                use_container_width=True
            )

        with c_exp2:
            st.subheader("📊 Extraction Report CSV")
            df_export = ExportService.generate_field_extraction_dataframe(
                docs,
                mask_sensitive=st.session_state.mask_sensitive_data
            )
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                "📥 Download Field Extraction CSV",
                data=csv_data,
                file_name="field_extraction_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.divider()
        st.subheader("📄 Individual Document JSONs")
        for d in docs:
            doc_json = ExportService.export_single_document_json(d)
            st.download_button(
                f"📥 Download {d.document_id} ({d.filename}) JSON",
                data=doc_json,
                file_name=f"{d.document_id}_{d.document_type.value}.json",
                mime="application/json"
            )

# ==============================================================================
# TAB 9: CONFIGURATION
# ==============================================================================
with tabs[8]:
    st.header("Processing Configuration")
    st.markdown("Configure multimodal vision keys, OCR engine prioritization, and privacy parameters.")

    st.subheader("🔑 Multimodal Vision Engine Settings")
    input_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Google Gemini API key for multimodal vision, handwriting OCR, and complex layout analysis."
    )
    if st.button("Update API Key"):
        st.session_state.gemini_api_key = input_key
        # Re-initialize client
        gemini_client = GeminiVisionClient(api_key=input_key)
        st.success("API key updated successfully.")
        st.rerun()

    st.divider()
    st.subheader("⚖️ Operational Confidence Threshold Rationale")
    st.markdown("""
    > **Why 0.85?**  
    > In financial, identity, and insurance operations (KYC, ECS mandates, tax disclosures), erroneous extractions carry high compliance and financial liability. A 0.85 baseline ensures that any field with degraded ink strokes, ambiguous date format (DD/MM vs MM/DD), or multiple checkbox markings is systematically routed to a human reviewer for visual confirmation.
    >  
    > *Note: 0.85 represents an operational baseline that should be calibrated against empirical ground-truth error curves.*
    """)

    st.divider()
    st.subheader("🔒 Privacy and Local Processing Disclosures")
    st.markdown("""
    - Documents are processed locally on the host machine using PyMuPDF and OpenCV by default.
    - Sensitive fields (Aadhaar, PAN, Bank Account Numbers) are masked in all audit summaries and UI tables unless explicitly unmasked.
    - If the external Gemini Vision API is configured, images are transmitted securely over TLS and are never stored or used for retraining.
    """)
