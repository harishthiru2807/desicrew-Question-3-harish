"""
Unit tests for domain-specific validators:
- Aadhaar (12 digits, Verhoeff checksum)
- PAN (5 letters, 4 digits, 1 letter)
- Driving Licence
- Passport number & Passport MRZ Line 2 (ICAO 9303 checksums)
- NACH Bank Account & IFSC code
- Dates (ambiguity detection, DD/MM vs MM/DD)
- Checkboxes
"""

import pytest
from services.validators import FieldValidator, validate_verhoeff, validate_mrz_check_digit

def test_aadhaar_validation():
    # Valid Aadhaar 12-digit format
    is_valid, status, err, norm = FieldValidator.validate_aadhaar_number("5482 9104 3821")
    assert is_valid is True
    assert norm == "5482 9104 3821"

    # Too short
    is_valid, status, err, _ = FieldValidator.validate_aadhaar_number("1234 5678")
    assert is_valid is False
    assert status == "invalid"

    # Missing
    is_valid, status, err, _ = FieldValidator.validate_aadhaar_number("")
    assert is_valid is False
    assert status == "missing"

def test_pan_validation():
    # Valid PAN
    is_valid, status, err, norm = FieldValidator.validate_pan_number("ABCPS1234F")
    assert is_valid is True
    assert status == "valid"
    assert norm == "ABCPS1234F"

    # Invalid characters / format
    is_valid, status, err, _ = FieldValidator.validate_pan_number("12345ABCDE")
    assert is_valid is False
    assert status == "invalid"

def test_dl_validation():
    # Valid DL
    is_valid, status, err, norm = FieldValidator.validate_dl_number("MH1220180054321")
    assert is_valid is True
    assert norm == "MH1220180054321"

    # Invalid DL
    is_valid, status, err, _ = FieldValidator.validate_dl_number("123")
    assert is_valid is False

def test_passport_and_mrz_validation():
    # Passport Number
    is_valid, status, err, norm = FieldValidator.validate_passport_number("Z2194832")
    assert is_valid is True
    assert norm == "Z2194832"

    # Valid MRZ Line 2 (44 characters with verified ICAO 9303 check digits)
    # Z2194832< (check 2), DOB 950510 (check 6), Expiry 350509 (check 0)
    valid_mrz = "Z2194832<2IND9505106M3505090<<<<<<<<<<<<<<02"
    is_valid, status, err, norm = FieldValidator.validate_mrz_line_2(valid_mrz)
    assert is_valid is True
    assert norm == valid_mrz

    # Invalid checksum detection
    bad_checksum_mrz = "Z2194832<9IND9505109M3505099<<<<<<<<<<<<<<02"
    is_valid, status, err, _ = FieldValidator.validate_mrz_line_2(bad_checksum_mrz)
    assert is_valid is False
    assert "failures" in err

    # Wrong length
    is_valid, status, err, _ = FieldValidator.validate_mrz_line_2("SHORTMRZ")
    assert is_valid is False

def test_ifsc_and_account_validation():
    # Valid IFSC
    is_valid, status, err, norm = FieldValidator.validate_ifsc_code("HDFC0001234")
    assert is_valid is True
    assert norm == "HDFC0001234"

    # 5th char not '0'
    is_valid, status, err, _ = FieldValidator.validate_ifsc_code("HDFC1001234")
    assert is_valid is False
    assert "must be '0'" in err

    # Valid Account Number (9 to 18 digits)
    is_valid, status, err, norm = FieldValidator.validate_bank_account_number("987654321012")
    assert is_valid is True
    assert norm == "987654321012"

    # Account with non-digits
    is_valid, status, err, _ = FieldValidator.validate_bank_account_number("98765A321012")
    assert is_valid is False

def test_date_ambiguity():
    # Unambiguous date (day > 12): 25/08/2024 -> normalized to 2024-08-25
    is_valid, status, err, norm = FieldValidator.validate_date("25/08/2024")
    assert is_valid is True
    assert status == "valid"
    assert norm == "2024-08-25"

    # Ambiguous date (both day and month <= 12): 03/04/2025
    is_valid, status, err, norm = FieldValidator.validate_date("03/04/2025")
    assert is_valid is True
    assert status == "ambiguous_date"
    assert "Ambiguous date" in err
