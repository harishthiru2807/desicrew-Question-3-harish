"""
Field Validators Module
Provides domain-specific validators for Indian identity, banking, and insurance documents.
Implements Verhoeff algorithm for Aadhaar, ICAO 9303 check digits for Passport MRZ Line 2,
strict PAN/IFSC regexes, date ambiguity analysis, and checkbox validation.
"""

import re
from datetime import datetime
from typing import Tuple, Optional, Any, Dict

# Verhoeff algorithm multiplication and permutation tables for Aadhaar
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

def validate_verhoeff(num_str: str) -> bool:
    """Validate 12-digit Aadhaar number using Verhoeff algorithm."""
    c = 0
    clean_num = "".join(ch for ch in num_str if ch.isdigit())
    if len(clean_num) != 12:
        return False
    for i, item in enumerate(reversed(clean_num)):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(item)]]
    return c == 0

def validate_mrz_check_digit(data_str: str, check_digit_char: str) -> bool:
    """ICAO 9303 MRZ check digit validation with 7-3-1 weight pattern."""
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(data_str):
        if ch == '<':
            val = 0
        elif ch.isdigit():
            val = int(ch)
        elif ch.isalpha():
            val = ord(ch.upper()) - 55  # A=10, B=11 ... Z=35
        else:
            val = 0
        total += val * weights[i % 3]
    expected = str(total % 10)
    return expected == check_digit_char

class FieldValidator:
    """Comprehensive validator for all 10 document types."""

    @staticmethod
    def validate_aadhaar_number(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not value or str(value).strip() == "":
            return False, "missing", "Aadhaar number is missing", None
        clean = re.sub(r"[\s-]", "", str(value).strip())
        if not clean.isdigit() or len(clean) != 12:
            return False, "invalid", f"Aadhaar number must be exactly 12 digits, got {len(clean)}", None
        if clean[0] in ['0', '1']:
            return False, "invalid", "Aadhaar number cannot start with 0 or 1", None
        
        # Format as 1234 5678 9012
        formatted = f"{clean[:4]} {clean[4:8]} {clean[8:]}"
        # Check Verhoeff if valid
        if not validate_verhoeff(clean):
            # Flag with warning but keep formatted
            return True, "valid_with_warning", "Valid 12 digits, but Verhoeff checksum test flagged", formatted
        return True, "valid", None, formatted

    @staticmethod
    def validate_pan_number(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not value or str(value).strip() == "":
            return False, "missing", "PAN number is missing", None
        clean = str(value).strip().upper().replace(" ", "")
        pan_regex = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
        if not re.match(pan_regex, clean):
            return False, "invalid", f"Invalid PAN format: '{clean}'. Expected 5 letters, 4 digits, 1 letter", None
        return True, "valid", None, clean

    @staticmethod
    def validate_dl_number(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not value or str(value).strip() == "":
            return False, "missing", "Driving licence number is missing", None
        clean = str(value).strip().upper().replace("-", "").replace(" ", "")
        # Indian DL format: 2 letters state code + 2 digits RTO + 11 digits (e.g. MH1220190012345)
        # or state code + digits (total 13-16 alphanumeric)
        if len(clean) < 10 or len(clean) > 20:
            return False, "invalid", f"DL number length unexpected ({len(clean)} characters)", None
        if not clean[:2].isalpha():
            return False, "invalid", "DL number must begin with a 2-letter state code", None
        return True, "valid", None, clean

    @staticmethod
    def validate_passport_number(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not value or str(value).strip() == "":
            return False, "missing", "Passport number is missing", None
        clean = str(value).strip().upper().replace(" ", "")
        # Indian passport: 1 letter followed by 7 digits
        passport_regex = r"^[A-PR-WYa-pr-wy][1-9][0-9]{6}$"
        if re.match(passport_regex, clean):
            return True, "valid", None, clean
        if len(clean) == 8 and clean[0].isalpha() and clean[1:].isdigit():
            return True, "valid", None, clean
        return False, "invalid", f"Invalid passport number format: '{clean}'", None

    @staticmethod
    def validate_mrz_line_2(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not value or str(value).strip() == "":
            return False, "missing", "MRZ Line 2 is missing", None
        clean = str(value).strip().upper().replace(" ", "")
        if len(clean) != 44:
            return False, "invalid", f"MRZ Line 2 must be exactly 44 characters, got {len(clean)}", clean
        
        # MRZ Line 2 format:
        # P# (0-8), Check (9), Nat (10-12), DOB (13-18), Check (19), Sex (20), Expiry (21-26), Check (27)
        doc_num = clean[0:9]
        doc_check = clean[9]
        dob = clean[13:19]
        dob_check = clean[19]
        expiry = clean[21:27]
        exp_check = clean[27]

        warnings = []
        if not validate_mrz_check_digit(doc_num, doc_check):
            warnings.append("Document number check digit mismatch")
        if not validate_mrz_check_digit(dob, dob_check):
            warnings.append("DOB check digit mismatch")
        if not validate_mrz_check_digit(expiry, exp_check):
            warnings.append("Expiry check digit mismatch")

        if warnings:
            return False, "invalid", f"MRZ Checksum failures: {'; '.join(warnings)}", clean
        return True, "valid", None, clean

    @staticmethod
    def validate_ifsc_code(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not value or str(value).strip() == "":
            return False, "missing", "IFSC code is missing", None
        clean = str(value).strip().upper().replace(" ", "")
        if len(clean) != 11:
            return False, "invalid", f"IFSC code must be exactly 11 characters, got {len(clean)}", None
        if not clean[:4].isalpha():
            return False, "invalid", "First 4 characters of IFSC must be alphabetic bank code", None
        if clean[4] != '0':
            return False, "invalid", f"5th character of IFSC must be '0', got '{clean[4]}'", None
        if not clean[5:].isalnum():
            return False, "invalid", "Last 6 characters of IFSC must be alphanumeric", None
        return True, "valid", None, clean

    @staticmethod
    def validate_bank_account_number(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not value or str(value).strip() == "":
            return False, "missing", "Bank account number is missing", None
        clean = re.sub(r"[\s-]", "", str(value).strip())
        if not clean.isdigit():
            return False, "invalid", "Bank account number must contain digits only", None
        if len(clean) < 9 or len(clean) > 18:
            return False, "invalid", f"Bank account number length must be 9-18 digits, got {len(clean)}", None
        return True, "valid", None, clean

    @staticmethod
    def validate_amount(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not value or str(value).strip() == "":
            return False, "missing", "Amount is missing", None
        # Clean currency symbols, commas, trailing slash-hyphen
        clean = re.sub(r"[₹Rs\.,/-]", "", str(value).strip())
        digits_only = "".join(ch for ch in str(value) if ch.isdigit() or ch == '.')
        try:
            val_float = float(digits_only)
            if val_float <= 0:
                return False, "invalid", "Amount must be greater than zero", None
            return True, "valid", None, f"{val_float:,.2f}"
        except Exception:
            return False, "invalid", f"Cannot parse '{value}' as numerical amount", None

    @staticmethod
    def validate_frequency(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not value or str(value).strip() == "":
            return False, "missing", "Debit frequency is missing", None
        val_lower = str(value).strip().lower()
        valid_frequencies = {
            "monthly": "Monthly",
            "quarterly": "Quarterly",
            "half-yearly": "Half-Yearly",
            "half yearly": "Half-Yearly",
            "semi-annually": "Half-Yearly",
            "yearly": "Yearly",
            "annual": "Yearly",
            "annually": "Yearly",
            "as and when presented": "As and when presented",
            "as & when presented": "As and when presented",
            "adhoc": "As and when presented"
        }
        for k, normalized in valid_frequencies.items():
            if k in val_lower:
                return True, "valid", None, normalized
        return False, "invalid", f"Unknown frequency '{value}'. Expected Monthly, Quarterly, Half-Yearly, Yearly, or As & when presented", None

    @staticmethod
    def validate_date(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """
        Validates date and detects ambiguity (e.g. 03/04/2025 where both day and month <= 12).
        Normalizes unambiguous dates to ISO YYYY-MM-DD.
        """
        if not value or str(value).strip() == "":
            return False, "missing", "Date is missing", None
        val_str = str(value).strip()

        # Check for year only (some Aadhaar cards state 'Year of Birth: 1985')
        if re.match(r"^(19|20)\d{2}$", val_str):
            return True, "valid", None, val_str

        # Patterns like DD/MM/YYYY or MM/DD/YYYY
        match = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$", val_str)
        if match:
            p1 = int(match.group(1))
            p2 = int(match.group(2))
            year = int(match.group(3))

            # Ambiguity check: if both p1 <= 12 and p2 <= 12 and p1 != p2
            if p1 <= 12 and p2 <= 12 and p1 != p2:
                # Ambiguous: could be DD/MM/YYYY or MM/DD/YYYY
                # Default to Indian convention (DD/MM/YYYY) but flag ambiguity
                try:
                    norm = datetime(year, p2, p1).strftime("%Y-%m-%d")
                    return True, "ambiguous_date", f"Ambiguous date '{val_str}' (could be {p1:02d}/{p2:02d} or {p2:02d}/{p1:02d})", norm
                except ValueError:
                    pass

            # Unambiguous: if p1 > 12 and p2 <= 12 -> DD/MM/YYYY
            if p1 > 12 and p2 <= 12:
                try:
                    dt = datetime(year, p2, p1)
                    return True, "valid", None, dt.strftime("%Y-%m-%d")
                except ValueError as e:
                    return False, "invalid", f"Invalid day/month: {e}", None

            # Unambiguous: if p2 > 12 and p1 <= 12 -> MM/DD/YYYY
            if p2 > 12 and p1 <= 12:
                try:
                    dt = datetime(year, p1, p2)
                    return True, "valid", None, dt.strftime("%Y-%m-%d")
                except ValueError as e:
                    return False, "invalid", f"Invalid day/month: {e}", None

        # Try ISO format YYYY-MM-DD
        iso_match = re.match(r"^(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})$", val_str)
        if iso_match:
            try:
                y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
                dt = datetime(y, m, d)
                return True, "valid", None, dt.strftime("%Y-%m-%d")
            except ValueError as e:
                return False, "invalid", f"Invalid date components: {e}", None

        # Named month formats: e.g. 15 Aug 1990 or August 15, 1990
        for fmt in ["%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y"]:
            try:
                dt = datetime.strptime(val_str, fmt)
                return True, "valid", None, dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return False, "invalid", f"Unrecognized date format: '{val_str}'", val_str

    @staticmethod
    def validate_checkbox(value: Optional[str]) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """Validate checkbox extraction for Multiple Policies Consent Form."""
        if not value or str(value).strip() == "":
            return False, "missing", "No checkbox selection detected", None
        val_str = str(value).strip()
        
        # Check for ambiguous or multiple selections
        if "multiple" in val_str.lower() or "ambiguous" in val_str.lower() or ";" in val_str or "," in val_str:
            return False, "ambiguous_checkbox", "Multiple or ambiguous checkboxes marked", val_str
        
        return True, "valid", None, val_str

    @staticmethod
    def validate_generic_text(value: Optional[str], field_label: str = "Field") -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not value or str(value).strip() == "" or str(value).strip().lower() in ["null", "none", "n/a", "not readable", "unreadable"]:
            return False, "missing", f"{field_label} is missing or unreadable", None
        val_str = str(value).strip()
        if len(val_str) < 2:
            return False, "invalid", f"{field_label} is too short ({len(val_str)} characters)", val_str
        return True, "valid", None, val_str
