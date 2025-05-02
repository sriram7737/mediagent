# mediagent_pipeline.py  

# mediagent_pipeline.py

# === 1. OCR Extraction ===
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()


def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text

# === 2. Lightweight Medical Info Extraction ===
import re

def extract_medical_info(text):
    diagnosis_pattern = re.findall(r"(?:Diagnosis|Dx)[:\-\s]*([\w\s,]+)", text, re.IGNORECASE)
    medication_pattern = re.findall(r"\b([A-Z][a-zA-Z]+\s\d+(?:mg|ml|MG|ML))\b", text)
    instructions_pattern = re.findall(r"Take[^.]*\.", text, re.IGNORECASE)

    return {
        "diagnosis": list(set(diagnosis_pattern)),
        "medications": list(set(medication_pattern)),
        "instructions": list(set(instructions_pattern))
    }

# === 3. LLM Explanation using Mistral (with Hugging Face Token) ===
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
hf_token = os.getenv("HF_TOKEN")  # Store your token as an environment variable

if hf_token is None:
    raise ValueError("Hugging Face token not found. Please set HF_TOKEN environment variable.")

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    use_auth_token=hf_token
)

def get_medical_explanation(diagnosis, medications):
    prompt = (
        f"You are a helpful medical assistant.\n"
        f"Explain the diagnosis: {diagnosis}.\n"
        f"Medications prescribed: {medications}.\n"
        f"Also provide general lifestyle advice."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === 4. Orchestrator ===
def process_prescription(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    med_info = extract_medical_info(text)
    explanation = get_medical_explanation(med_info["diagnosis"], med_info["medications"])
    return {
        "raw_text": text,
        "medical_info": med_info,
        "explanation": explanation
    }

# === Example ===
if __name__ == "__main__":
    result = process_prescription("sample_prescriptions/prescription1.pdf")
    print("\nExtracted Info:", result["medical_info"])
    print("\nExplanation:\n", result["explanation"])
