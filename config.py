"""
Application Configuration Module
Loads environment variables, default paths, and document specifications.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BASE_DIR / "configs"
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
EXPORTS_DIR = BASE_DIR / "exports"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_documents"

# Ensure directories exist
for path in [UPLOAD_DIR, PROCESSED_DIR, EXPORTS_DIR, SAMPLE_DOCS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.85"))
ENABLE_SENSITIVE_DATA_MASKING = os.getenv("ENABLE_SENSITIVE_DATA_MASKING", "True").lower() == "true"
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# Load Document Definitions
CONFIG_FILE = CONFIGS_DIR / "document_types.json"
DOCUMENT_TYPES = {}
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        DOCUMENT_TYPES = data.get("document_types", {})

def get_document_type_names():
    """Return dictionary of doc_type_id -> human readable display name."""
    return {k: v.get("name", k) for k, v in DOCUMENT_TYPES.items()}

def get_document_fields(doc_type_id: str):
    """Return list of field configuration dicts for a given document type."""
    return DOCUMENT_TYPES.get(doc_type_id, {}).get("required_fields", [])

def get_doc_type_by_display_name(display_name: str) -> str:
    """Return doc_type_id from human readable display name."""
    for doc_id, doc_meta in DOCUMENT_TYPES.items():
        if doc_meta.get("name", "").lower() == display_name.lower():
            return doc_id
    return ""
