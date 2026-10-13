"""
Gemini Vision Service
Handles multimodal vision extraction with Gemini 3.8 / 2.5 Flash.
Gracefully catches 429 ResourceExhausted, quota limits, or network errors,
reporting status clearly to enable automatic fallback to local engines.
"""

import os
import json
import re
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME

class GeminiVisionClient:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or "models/gemini-3.8-flash"
        self._is_available = False
        self._last_error = None
        self._init_client()

    def _init_client(self):
        if not self.api_key or self.api_key.strip() == "":
            self._is_available = False
            self._last_error = "API key not provided"
            return
        try:
            genai.configure(api_key=self.api_key)
            self._is_available = True
            self._last_error = None
        except Exception as e:
            self._is_available = False
            self._last_error = str(e)

    def is_available(self) -> Tuple[bool, Optional[str]]:
        return self._is_available, self._last_error

    def analyze_document(
        self,
        image_path: str,
        target_doc_type: Optional[str] = None,
        expected_fields: Optional[list] = None
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Sends the document image to Gemini Vision with structured prompts.
        Returns: (success, result_dict, error_message)
        """
        if not self._is_available:
            return False, {}, self._last_error or "Gemini client not initialized"

        try:
            pil_img = Image.open(image_path)
            model = genai.GenerativeModel(self.model_name)

            fields_hint = ""
            if expected_fields:
                fields_hint = f"Focus on extracting these fields: {', '.join(expected_fields)}."

            prompt = f"""
You are an expert AI Document Processing & Handwriting Extraction system.
Analyze this document image carefully.

Document Type Classification options:
- aadhaar_card
- pan_card
- driving_licence
- passport
- nach_ecs_mandate
- fatca_annexure_form
- benefit_illustration_declaration
- moral_hazard_questionnaire
- multiple_policies_consent_form
- suitability_profiler_declaration
- unsupported_ambiguous

{f"Note: This document is expected to be '{target_doc_type}'." if target_doc_type else ""}
{fields_hint}

For handwritten forms:
- Carefully transcribe handwritten pen strokes for names, numbers, dates, places, IFSC, and amounts.
- For checkboxes, detect which checkbox is marked (selected vs unselected).
- For passports, extract MRZ Line 2 (44 characters).

Return STRICT JSON ONLY matching this format:
{{
  "document_type": "<predicted_doc_type_key>",
  "classification_confidence": 0.95,
  "classification_reasons": ["Visible header Income Tax Department", "Permanent Account Number Card format"],
  "full_text": "<all visible printed and handwritten text>",
  "fields": {{
    "<field_name>": {{
      "value": "<extracted value or null if unreadable>",
      "raw_text": "<original raw text observed>",
      "confidence": 0.90,
      "source_type": "handwritten" | "printed" | "checkbox" | "machine_readable_zone",
      "bounding_box": [ymin, xmin, ymax, xmax]  // relative 0-1000 scale
    }}
  }}
}}
Do not include markdown triple backticks around the json. Output pure JSON only.
"""

            response = model.generate_content([prompt, pil_img])
            text = response.text.strip()
            
            # Clean markdown code blocks if present
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
                
            parsed = json.loads(text)
            return True, parsed, None

        except Exception as e:
            err_msg = str(e)
            if "ResourceExhausted" in err_msg or "429" in err_msg or "quota" in err_msg.lower():
                self._last_error = "Gemini Free-Tier Quota Exceeded (429). Using Local Fallback Engine."
            else:
                self._last_error = f"Gemini API Error: {err_msg}"
            return False, {}, self._last_error
