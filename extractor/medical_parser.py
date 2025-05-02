# extractor/medical_parser.py

import re

def extract_medical_info(text):
    """
    Extracts diagnosis, medications, and instructions using regex patterns.
    Returns a dictionary with extracted fields.
    """
    diagnosis_pattern = re.findall(r"(?:Diagnosis|Dx)[:\-\s]*([\w\s,]+)", text, re.IGNORECASE)
    medication_pattern = re.findall(r"\b([A-Z][a-zA-Z]+\s\d+(?:mg|ml|MG|ML))\b", text)
    instructions_pattern = re.findall(r"Take[^.]*\.", text, re.IGNORECASE)

    return {
        "diagnosis": list(set(diagnosis_pattern)),
        "medications": list(set(medication_pattern)),
        "instructions": list(set(instructions_pattern))
    }
