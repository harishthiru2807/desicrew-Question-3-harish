"""
Ground-Truth Evaluation Service
Evaluates model predictions against user-supplied ground-truth answer keys (JSON or CSV).
Calculates exact-match metrics, printed vs handwritten breakdowns, high-risk field metrics,
missing rates, false acceptance rates, and review precision/recall.
"""

import json
import re
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from schemas.document_schema import DocumentRecord

class EvaluationService:
    @staticmethod
    def normalize_for_comparison(val: Any, field_type: str = "text") -> str:
        """
        Field-specific normalization preserving semantic identity:
        - Numbers & IDs: strip spaces, hyphens, uppercase
        - Dates: standardize separators
        - General text: whitespace collapse, lowercase
        """
        if val is None:
            return ""
        s = str(val).strip()
        if s.lower() in ["null", "none", "n/a", ""]:
            return ""

        if field_type in ["pan_number", "tin_pan", "ifsc_code", "passport_number", "dl_number"]:
            return s.upper().replace(" ", "").replace("-", "")
        elif field_type in ["aadhaar_number", "bank_account_number"]:
            return re.sub(r"[\s-]", "", s)
        elif "date" in field_type:
            # normalize 01/02/2020 vs 01-02-2020 vs 2020-02-01
            return s.replace("/", "-").replace(".", "-")
        else:
            # General text
            return re.sub(r"\s+", " ", s).lower().strip()

    @classmethod
    def evaluate_batch(
        cls,
        predicted_docs: List[DocumentRecord],
        ground_truth_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compares predicted documents against ground truth data.
        Ground truth can be keyed by document_id or filename.
        """
        total_docs = len(predicted_docs)
        if total_docs == 0:
            return {"error": "No documents provided for evaluation"}

        doc_classification_matches = 0
        docs_evaluated = 0

        # Field level accumulators
        total_fields_evaluated = 0
        exact_matches_total = 0
        missing_fields = 0

        printed_total = 0
        printed_matches = 0

        handwritten_total = 0
        handwritten_matches = 0

        # High-risk target categories
        high_risk_categories = {
            "ifsc": {"total": 0, "match": 0},
            "pan": {"total": 0, "match": 0},
            "bank_account": {"total": 0, "match": 0},
            "dates": {"total": 0, "match": 0},
            "places": {"total": 0, "match": 0}
        }

        # Field-by-field accuracy tracking
        per_field_stats: Dict[str, Dict[str, int]] = {}
        comparison_rows: List[Dict[str, Any]] = []

        # Error and Review tracking
        false_acceptances = 0   # Incorrect extraction that had high confidence (>= threshold) and was NOT flagged
        true_pos_flagged = 0    # Actually incorrect/missing and flagged
        false_pos_flagged = 0   # Actually correct but flagged
        false_neg_unflagged = 0 # Actually incorrect but NOT flagged
        true_neg_unflagged = 0  # Actually correct and NOT flagged

        for doc in predicted_docs:
            gt_doc = ground_truth_data.get(doc.document_id) or ground_truth_data.get(doc.filename)
            if not gt_doc:
                continue

            docs_evaluated += 1

            # 1. Document classification match
            expected_type = gt_doc.get("document_type") or gt_doc.get("type")
            if expected_type:
                pred_type = doc.document_type.value
                if cls.normalize_for_comparison(pred_type) == cls.normalize_for_comparison(expected_type):
                    doc_classification_matches += 1

            # 2. Field evaluations
            gt_fields = gt_doc.get("fields", {})
            for fname, f_pred in doc.fields.items():
                if fname not in gt_fields:
                    continue

                gt_val = gt_fields[fname]
                pred_val = f_pred.reviewed_value if f_pred.review_status == "edited" and f_pred.reviewed_value is not None else f_pred.value

                norm_gt = cls.normalize_for_comparison(gt_val, field_type=fname)
                norm_pred = cls.normalize_for_comparison(pred_val, field_type=fname)

                is_match = (norm_gt != "") and (norm_gt == norm_pred)
                is_missing = (norm_pred == "")

                total_fields_evaluated += 1
                if is_match:
                    exact_matches_total += 1
                if is_missing:
                    missing_fields += 1

                # Source type breakdown
                if f_pred.source_type == "handwritten":
                    handwritten_total += 1
                    if is_match:
                        handwritten_matches += 1
                else:
                    printed_total += 1
                    if is_match:
                        printed_matches += 1

                # High-risk groups
                if fname == "ifsc_code":
                    high_risk_categories["ifsc"]["total"] += 1
                    if is_match: high_risk_categories["ifsc"]["match"] += 1
                elif fname in ["pan_number", "tin_pan"]:
                    high_risk_categories["pan"]["total"] += 1
                    if is_match: high_risk_categories["pan"]["match"] += 1
                elif fname == "bank_account_number":
                    high_risk_categories["bank_account"]["total"] += 1
                    if is_match: high_risk_categories["bank_account"]["match"] += 1
                elif "date" in fname:
                    high_risk_categories["dates"]["total"] += 1
                    if is_match: high_risk_categories["dates"]["match"] += 1
                elif fname in ["place", "place_of_birth"]:
                    high_risk_categories["places"]["total"] += 1
                    if is_match: high_risk_categories["places"]["match"] += 1

                # Per-field breakdown
                if fname not in per_field_stats:
                    per_field_stats[fname] = {"total": 0, "match": 0}
                per_field_stats[fname]["total"] += 1
                if is_match:
                    per_field_stats[fname]["match"] += 1

                # Flagging evaluation
                # Error condition: not a match
                has_error = not is_match
                was_flagged = f_pred.needs_review

                if has_error and was_flagged:
                    true_pos_flagged += 1
                elif not has_error and was_flagged:
                    false_pos_flagged += 1
                elif has_error and not was_flagged:
                    false_neg_unflagged += 1
                    false_acceptances += 1
                elif not has_error and not was_flagged:
                    true_neg_unflagged += 1

                comparison_rows.append({
                    "Document ID": doc.document_id,
                    "Field Name": f_pred.display_name or fname,
                    "Predicted Value": pred_val,
                    "Ground Truth": gt_val,
                    "Exact Match": "Yes" if is_match else "No",
                    "Confidence": f_pred.confidence,
                    "Flagged for Review": "Yes" if was_flagged else "No",
                    "Source Type": f_pred.source_type
                })

        if docs_evaluated == 0:
            return {"error": "No matching document IDs or filenames found between predictions and ground truth."}

        # Calculate metrics
        classification_accuracy = (doc_classification_matches / docs_evaluated) * 100.0
        overall_field_accuracy = (exact_matches_total / total_fields_evaluated * 100.0) if total_fields_evaluated > 0 else 0.0
        printed_accuracy = (printed_matches / printed_total * 100.0) if printed_total > 0 else 0.0
        handwritten_accuracy = (handwritten_matches / handwritten_total * 100.0) if handwritten_total > 0 else 0.0
        missing_rate = (missing_fields / total_fields_evaluated * 100.0) if total_fields_evaluated > 0 else 0.0
        false_acceptance_rate = (false_acceptances / total_fields_evaluated * 100.0) if total_fields_evaluated > 0 else 0.0
        flagging_rate = ((true_pos_flagged + false_pos_flagged) / total_fields_evaluated * 100.0) if total_fields_evaluated > 0 else 0.0

        # Review precision and recall
        actual_errors = true_pos_flagged + false_neg_unflagged
        predicted_errors = true_pos_flagged + false_pos_flagged
        review_precision = (true_pos_flagged / predicted_errors * 100.0) if predicted_errors > 0 else 100.0
        review_recall = (true_pos_flagged / actual_errors * 100.0) if actual_errors > 0 else 100.0

        field_accuracy_table = []
        for fn, st in per_field_stats.items():
            tot = st["total"]
            m = st["match"]
            acc = round((m / tot * 100.0), 1) if tot > 0 else 0.0
            field_accuracy_table.append({
                "Field": fn,
                "Total Evaluated": tot,
                "Matches": m,
                "Accuracy (%)": acc
            })

        high_risk_table = []
        for grp, st in high_risk_categories.items():
            tot = st["total"]
            m = st["match"]
            acc = round((m / tot * 100.0), 1) if tot > 0 else 0.0
            high_risk_table.append({
                "Group": grp.upper(),
                "Evaluated": tot,
                "Matches": m,
                "Accuracy (%)": acc
            })

        return {
            "docs_evaluated": docs_evaluated,
            "classification_accuracy_pct": round(classification_accuracy, 2),
            "overall_field_accuracy_pct": round(overall_field_accuracy, 2),
            "printed_accuracy_pct": round(printed_accuracy, 2),
            "handwritten_accuracy_pct": round(handwritten_accuracy, 2),
            "missing_field_rate_pct": round(missing_rate, 2),
            "false_acceptance_rate_pct": round(false_acceptance_rate, 2),
            "flagging_rate_pct": round(flagging_rate, 2),
            "review_precision_pct": round(review_precision, 2),
            "review_recall_pct": round(review_recall, 2),
            "field_accuracy_breakdown": field_accuracy_table,
            "high_risk_breakdown": high_risk_table,
            "comparison_df": pd.DataFrame(comparison_rows)
        }

    @staticmethod
    def parse_ground_truth_file(content: str, filename: str) -> Dict[str, Any]:
        """Parses uploaded ground-truth JSON or CSV into standard dict format."""
        if filename.endswith(".json"):
            data = json.loads(content)
            # Normalize list into dict keyed by document_id or filename
            if isinstance(data, list):
                result = {}
                for item in data:
                    key = item.get("document_id") or item.get("filename")
                    if key:
                        result[key] = item
                return result
            elif isinstance(data, dict):
                return data
        elif filename.endswith(".csv"):
            import io
            df = pd.read_csv(io.StringIO(content))
            result = {}
            for _, row in df.iterrows():
                doc_key = str(row.get("document_id") or row.get("filename") or "")
                if not doc_key:
                    continue
                if doc_key not in result:
                    result[doc_key] = {
                        "document_id": doc_key,
                        "document_type": row.get("document_type", ""),
                        "fields": {}
                    }
                fname = str(row.get("field_name") or "")
                val = str(row.get("expected_value") or row.get("ground_truth_value") or "")
                if fname:
                    result[doc_key]["fields"][fname] = val
            return result
        return {}
