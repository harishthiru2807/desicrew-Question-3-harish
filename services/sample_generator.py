"""
Sample Document Generator
Generates a realistic synthetic evaluation set of 10 mixed identity, banking,
and insurance documents (both PDFs and PNGs) along with matching Ground-Truth JSON and CSV keys.
Enables instant end-to-end evaluation and testing.
"""

import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
from config import SAMPLE_DOCS_DIR

GROUND_TRUTH_DATA = {
    "DOC_001": {
        "document_id": "DOC_001",
        "filename": "DOC_001_aadhaar_card.png",
        "document_type": "aadhaar_card",
        "fields": {
            "aadhaar_number": "5482 9104 3821",
            "full_name": "Aarav Sharma",
            "date_of_birth": "1992-08-14",
            "address": "Flat 402, Shanti Vihar, MG Road, Pune, Maharashtra 411001"
        }
    },
    "DOC_002": {
        "document_id": "DOC_002",
        "filename": "DOC_002_pan_card.png",
        "document_type": "pan_card",
        "fields": {
            "pan_number": "ABCPS1234F",
            "full_name": "PRIYA SUNDARAM",
            "fathers_name": "SUNDARAM RAMASWAMY",
            "date_of_birth": "1988-11-22"
        }
    },
    "DOC_003": {
        "document_id": "DOC_003",
        "filename": "DOC_003_driving_licence.png",
        "document_type": "driving_licence",
        "fields": {
            "dl_number": "MH1220180054321",
            "name": "VIKRAM JOSHI",
            "date_of_issue": "2018-03-15",
            "valid_till_date": "2038-03-14"
        }
    },
    "DOC_004": {
        "document_id": "DOC_004",
        "filename": "DOC_004_passport.pdf",
        "document_type": "passport",
        "fields": {
            "passport_number": "Z2194832",
            "date_of_birth": "1995-05-10",
            "date_of_expiry": "2035-05-09",
            "mrz_line_2": "Z2194832<2IND9505106M3505090<<<<<<<<<<<<<<02"
        }
    },
    "DOC_005": {
        "document_id": "DOC_005",
        "filename": "DOC_005_nach_ecs_mandate.pdf",
        "document_type": "nach_ecs_mandate",
        "fields": {
            "bank_account_number": "987654321012",
            "ifsc_code": "HDFC0001234",
            "bank_name": "HDFC Bank Limited",
            "amount_in_figures": "25,000.00",
            "frequency": "Monthly"
        }
    },
    "DOC_006": {
        "document_id": "DOC_006",
        "filename": "DOC_006_fatca_annexure_form.png",
        "document_type": "fatca_annexure_form",
        "fields": {
            "policy_number": "POL87654321",
            "tin_pan": "BLRPR8765K",
            "fathers_name": "Rajesh Kumar",
            "place_of_birth": "Bengaluru",
            "nationality": "Indian"
        }
    },
    "DOC_007": {
        "document_id": "DOC_007",
        "filename": "DOC_007_benefit_illustration_declaration.pdf",
        "document_type": "benefit_illustration_declaration",
        "fields": {
            "application_number": "APP44332211",
            "policyholder_name": "Sunita Patel",
            "date": "2024-04-12",
            "place": "Ahmedabad"
        }
    },
    "DOC_008": {
        "document_id": "DOC_008",
        "filename": "DOC_008_moral_hazard_questionnaire.png",
        "document_type": "moral_hazard_questionnaire",
        "fields": {
            "application_number": "APP99887766",
            "name_of_life_assured": "Rohan Malhotra",
            "nominee_relationship": "Spouse",
            "date": "2024-06-18",
            "place": "Chandigarh"
        }
    },
    "DOC_009": {
        "document_id": "DOC_009",
        "filename": "DOC_009_multiple_policies_consent_form.png",
        "document_type": "multiple_policies_consent_form",
        "fields": {
            "proposer_name": "Ananya Deshmukh",
            "reason_for_multiple_policies": "Family financial security",
            "date": "2024-08-25",
            "place": "Nagpur"
        }
    },
    "DOC_010": {
        "document_id": "DOC_010",
        "filename": "DOC_010_suitability_profiler_declaration.pdf",
        "document_type": "suitability_profiler_declaration",
        "fields": {
            "application_number": "APP55667788",
            "name_of_life_assured": "Karthik Iyer",
            "name_of_agent_sp": "Deepak Verma",
            "date": "2024-09-30",
            "place": "Chennai"
        }
    }
}

def create_synthetic_documents(output_dir: Path = SAMPLE_DOCS_DIR):
    """Generates all 10 evaluation documents and ground truth files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. DOC_001 Aadhaar
    img1 = Image.new("RGB", (900, 560), color="#FDFDFE")
    d1 = ImageDraw.Draw(img1)
    d1.rectangle([(20, 20), (880, 540)], outline="#E2E8F0", width=3)
    d1.rectangle([(30, 30), (870, 90)], fill="#FFF7ED")
    d1.text((50, 45), "GOVERNMENT OF INDIA", fill="#EA580C")
    d1.text((50, 65), "Unique Identification Authority of India (UIDAI) - Mera Aadhaar", fill="#1E293B")
    d1.text((50, 130), "To,\nName: Aarav Sharma\nAddress: Flat 402, Shanti Vihar, MG Road, Pune, Maharashtra 411001", fill="#0F172A")
    d1.text((50, 230), "DOB: 14/08/1992   Gender: Male", fill="#0F172A")
    d1.text((50, 320), "Aadhaar No:\n5482 9104 3821", fill="#1E3A8A")
    d1.text((50, 480), "My Aadhaar, My Identity - Enrolment No: 1024/99382/01928", fill="#64748B")
    img1.save(output_dir / "DOC_001_aadhaar_card.png")

    # 2. DOC_002 PAN Card
    img2 = Image.new("RGB", (900, 560), color="#F8FAFC")
    d2 = ImageDraw.Draw(img2)
    d2.rectangle([(20, 20), (880, 540)], outline="#CBD5E1", width=3)
    d2.rectangle([(30, 30), (870, 95)], fill="#EFF6FF")
    d2.text((50, 40), "INCOME TAX DEPARTMENT / GOVT. OF INDIA", fill="#1E40AF")
    d2.text((50, 65), "Permanent Account Number Card", fill="#1E3A8A")
    d2.text((50, 130), "Permanent Account Number / PAN:\nABCPS1234F", fill="#0F172A")
    d2.text((50, 220), "Name:\nPRIYA SUNDARAM", fill="#0F172A")
    d2.text((50, 310), "Father's Name:\nSUNDARAM RAMASWAMY", fill="#0F172A")
    d2.text((50, 400), "Date of Birth:\n22/11/1988", fill="#0F172A")
    img2.save(output_dir / "DOC_002_pan_card.png")

    # 3. DOC_003 Driving Licence
    img3 = Image.new("RGB", (900, 560), color="#FFFFFF")
    d3 = ImageDraw.Draw(img3)
    d3.rectangle([(20, 20), (880, 540)], outline="#E2E8F0", width=2)
    d3.text((50, 40), "UNION OF INDIA - TRANSPORT DEPARTMENT", fill="#0369A1")
    d3.text((50, 65), "FORM 7 - DRIVING LICENCE", fill="#0F172A")
    d3.text((50, 120), "DL No: MH1220180054321", fill="#0F172A")
    d3.text((50, 180), "Name: VIKRAM JOSHI", fill="#0F172A")
    d3.text((50, 240), "Date of Issue: 15/03/2018", fill="#0F172A")
    d3.text((50, 300), "Valid Till: 14/03/2038", fill="#0F172A")
    d3.text((50, 360), "Authorized to Drive: LMV-NT, MCWG", fill="#334155")
    img3.save(output_dir / "DOC_003_driving_licence.png")

    # 4. DOC_004 Passport (PDF)
    doc4 = fitz.open()
    p4 = doc4.new_page(width=612, height=792)
    p4.insert_text((50, 60), "REPUBLIC OF INDIA / PASSPORT", fontsize=16, color=(0.1, 0.2, 0.5))
    p4.insert_text((50, 90), "Type: P   Country Code: IND   Passport No: Z2194832", fontsize=12)
    p4.insert_text((50, 130), "Given Names: ROHIT", fontsize=12)
    p4.insert_text((50, 160), "Nationality: INDIAN", fontsize=12)
    p4.insert_text((50, 190), "Date of Birth: 10/05/1995   Sex: M", fontsize=12)
    p4.insert_text((50, 220), "Place of Issue: MUMBAI", fontsize=12)
    p4.insert_text((50, 250), "Date of Expiry: 09/05/2035", fontsize=12)
    # MRZ lines
    p4.insert_text((50, 680), "P<INDKUMAR<<ROHIT<<<<<<<<<<<<<<<<<<<<<<<<<<<", fontsize=13, fontname="courier")
    p4.insert_text((50, 710), "Z2194832<2IND9505106M3505090<<<<<<<<<<<<<<02", fontsize=13, fontname="courier")
    doc4.save(str(output_dir / "DOC_004_passport.pdf"))
    doc4.close()

    # 5. DOC_005 NACH / ECS Mandate (PDF)
    doc5 = fitz.open()
    p5 = doc5.new_page(width=612, height=792)
    p5.insert_text((50, 60), "NACH / ECS MANDATE FORM (NATIONAL AUTOMATED CLEARING HOUSE)", fontsize=14, color=(0.1, 0.2, 0.4))
    p5.insert_text((50, 90), "UMRN: 12345678901234567890   Sponsor Bank Code: HDFC0000001", fontsize=10)
    p5.insert_text((50, 130), "Utility Code: NACH000010000293   Debit Type: Maximum Amount", fontsize=10)
    p5.insert_text((50, 180), "Bank A/c No: 987654321012", fontsize=12)
    p5.insert_text((50, 220), "IFSC Code: HDFC0001234", fontsize=12)
    p5.insert_text((50, 260), "with Bank Name: HDFC Bank Limited", fontsize=12)
    p5.insert_text((50, 300), "Amount: Rs. 25,000.00 (Twenty Five Thousand Only)", fontsize=12)
    p5.insert_text((50, 340), "Frequency: Monthly", fontsize=12)
    p5.insert_text((50, 400), "I hereby authorize the above mandate debit to my account.", fontsize=10)
    doc5.save(str(output_dir / "DOC_005_nach_ecs_mandate.pdf"))
    doc5.close()

    # 6. DOC_006 FATCA Annexure Form (PNG)
    img6 = Image.new("RGB", (900, 650), color="#FFFFFF")
    d6 = ImageDraw.Draw(img6)
    d6.rectangle([(20, 20), (880, 630)], outline="#E2E8F0", width=2)
    d6.text((50, 40), "FATCA / CRS ANNEXURE FORM - TAX RESIDENCY DECLARATION", fill="#0F172A")
    d6.text((50, 70), "Foreign Account Tax Compliance Act (FATCA) Details", fill="#475569")
    d6.text((50, 130), "Policy No: POL87654321", fill="#1E293B")
    d6.text((50, 190), "TIN / PAN: BLRPR8765K", fill="#1E293B")
    d6.text((50, 250), "Father's Name: Rajesh Kumar", fill="#1E293B")
    d6.text((50, 310), "Place of Birth: Bengaluru", fill="#1E293B")
    d6.text((50, 370), "Country of Tax Residence: India   Nationality: Indian", fill="#1E293B")
    d6.text((50, 440), "I confirm that I am a tax resident of India and not of any other foreign country.", fill="#64748B")
    img6.save(output_dir / "DOC_006_fatca_annexure_form.png")

    # 7. DOC_007 Benefit Illustration Declaration (PDF)
    doc7 = fitz.open()
    p7 = doc7.new_page(width=612, height=792)
    p7.insert_text((50, 60), "BENEFIT ILLUSTRATION POLICYHOLDER DECLARATION", fontsize=15, color=(0.1, 0.3, 0.6))
    p7.insert_text((50, 90), "I confirm that I have understood the features, guaranteed and non-guaranteed", fontsize=10)
    p7.insert_text((50, 110), "projected investment returns as illustrated in the Benefit Illustration schedule.", fontsize=10)
    p7.insert_text((50, 160), "Application No: APP44332211", fontsize=12)
    p7.insert_text((50, 200), "Policyholder Name: Sunita Patel", fontsize=12)
    p7.insert_text((50, 240), "Date: 12/04/2024", fontsize=12)
    p7.insert_text((50, 280), "Place: Ahmedabad", fontsize=12)
    p7.insert_text((50, 350), "Signature of Policyholder: [Signed Sunita Patel]", fontsize=11)
    doc7.save(str(output_dir / "DOC_007_benefit_illustration_declaration.pdf"))
    doc7.close()

    # 8. DOC_008 Moral Hazard Questionnaire (PNG)
    img8 = Image.new("RGB", (900, 650), color="#FFFFFF")
    d8 = ImageDraw.Draw(img8)
    d8.rectangle([(20, 20), (880, 630)], outline="#E2E8F0", width=2)
    d8.text((50, 40), "CONFIDENTIAL MORAL HAZARD QUESTIONNAIRE (MHQ)", fill="#991B1B")
    d8.text((50, 70), "Special Confidential Report on Insurable Interest & Financial Status", fill="#475569")
    d8.text((50, 130), "Application No: APP99887766", fill="#0F172A")
    d8.text((50, 190), "Name of Life Assured: Rohan Malhotra", fill="#0F172A")
    d8.text((50, 250), "Nominee Relationship: Spouse", fill="#0F172A")
    d8.text((50, 310), "Date: 18/06/2024", fill="#0F172A")
    d8.text((50, 370), "Place: Chandigarh", fill="#0F172A")
    d8.text((50, 450), "Report Summary: Moral hazard is verified nil; insurable interest verified.", fill="#334155")
    img8.save(output_dir / "DOC_008_moral_hazard_questionnaire.png")

    # 9. DOC_009 Multiple Policies Consent Form (PNG)
    img9 = Image.new("RGB", (900, 650), color="#FFFFFF")
    d9 = ImageDraw.Draw(img9)
    d9.rectangle([(20, 20), (880, 630)], outline="#E2E8F0", width=2)
    d9.text((50, 40), "CONSENT FORM FOR MULTIPLE POLICIES", fill="#0F172A")
    d9.text((50, 70), "Declaration regarding existing insurance policies and additional coverage", fill="#475569")
    d9.text((50, 120), "Proposer Name: Ananya Deshmukh", fill="#0F172A")
    d9.text((50, 160), "Please select the primary reason for multiple policies:", fill="#0F172A")
    
    # Draw Checkboxes
    options = [
        "Tax planning / Investment growth",
        "Risk coverage / Life protection",
        "Family financial security",
        "Retirement planning"
    ]
    y_pos = 200
    for idx, opt in enumerate(options):
        # Draw checkbox square
        d9.rectangle([(70, y_pos), (95, y_pos + 25)], outline="#0F172A", width=2)
        if idx == 2:  # Option 3 "Family financial security" is marked
            d9.line([(73, y_pos + 12), (81, y_pos + 22)], fill="#1E40AF", width=3)
            d9.line([(81, y_pos + 22), (93, y_pos + 5)], fill="#1E40AF", width=3)
        d9.text((115, y_pos + 5), opt, fill="#1E293B")
        y_pos += 45

    d9.text((50, 410), "Date: 25/08/2024", fill="#0F172A")
    d9.text((50, 460), "Place: Nagpur", fill="#0F172A")
    img9.save(output_dir / "DOC_009_multiple_policies_consent_form.png")

    # 10. DOC_010 Suitability Profiler Declaration (PDF)
    doc10 = fitz.open()
    p10 = doc10.new_page(width=612, height=792)
    p10.insert_text((50, 60), "SUITABILITY PROFILER & NEED ANALYSIS DECLARATION", fontsize=15, color=(0.1, 0.3, 0.5))
    p10.insert_text((50, 90), "Declaration by Specified Person / Agent on Need Analysis Recommendation", fontsize=10)
    p10.insert_text((50, 140), "Application No: APP55667788", fontsize=12)
    p10.insert_text((50, 180), "Name of Life Assured: Karthik Iyer", fontsize=12)
    p10.insert_text((50, 220), "Name of Agent/SP: Deepak Verma", fontsize=12)
    p10.insert_text((50, 260), "Date: 30/09/2024", fontsize=12)
    p10.insert_text((50, 300), "Place: Chennai", fontsize=12)
    p10.insert_text((50, 370), "Recommendation verified in accordance with IRDAI suitability guidelines.", fontsize=10)
    doc10.save(str(output_dir / "DOC_010_suitability_profiler_declaration.pdf"))
    doc10.close()

    # Save Ground Truth JSON
    gt_json_path = output_dir / "ground_truth_answer_key.json"
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(list(GROUND_TRUTH_DATA.values()), f, indent=2)

    # Save Ground Truth CSV
    gt_csv_rows = []
    for doc_id, doc_data in GROUND_TRUTH_DATA.items():
        for fname, fval in doc_data["fields"].items():
            gt_csv_rows.append({
                "document_id": doc_id,
                "filename": doc_data["filename"],
                "document_type": doc_data["document_type"],
                "field_name": fname,
                "expected_value": fval
            })
    import pandas as pd
    gt_df = pd.DataFrame(gt_csv_rows)
    gt_df.to_csv(output_dir / "ground_truth_answer_key.csv", index=False)

    return True

if __name__ == "__main__":
    create_synthetic_documents()
    print("Successfully generated 10 evaluation documents and answer keys.")
