# app.py
import streamlit as st
from mediagent_pipeline import process_prescription
import os
import tempfile

st.set_page_config(page_title="MediAgent - Prescription Assistant", layout="centered")

# Custom HTML + CSS (optional logo and style)
st.markdown("""
    <style>
        .main-title { text-align: center; color: #2a9d8f; font-size: 40px; font-weight: bold; margin-top: 10px; }
        .section-title { font-size: 24px; color: #2a9d8f; margin-top: 20px; }
        .info-box { background-color: #f5f5f5; padding: 10px; border-radius: 8px; }
    </style>
    <div style='text-align:center;'>
        <img src='app/static/images/mediagent_logo.png' width='140'/>
    </div>
    <div class='main-title'>MediAgent - AI Medical Assistant 💊</div>
""", unsafe_allow_html=True)

st.markdown("""
Upload a scanned **prescription PDF or image**, and MediAgent will:
- Extract medical info (diagnosis, medications, instructions)
- Explain the diagnosis & meds using a local LLM (Mistral)
- Provide general lifestyle advice
""")

uploaded_file = st.file_uploader("Upload your prescription (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    with st.spinner("Processing your prescription with MediAgent..."):
        try:
            result = process_prescription(temp_file_path)
            st.success("Extraction complete! ✅")

            st.markdown("<div class='section-title'>🧾 Extracted Information</div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='info-box'>
                <b>Diagnosis:</b> {', '.join(result["medical_info"].get("diagnosis", []))}<br>
                <b>Medications:</b> {', '.join(result["medical_info"].get("medications", []))}<br>
                <b>Instructions:</b> {', '.join(result["medical_info"].get("instructions", []))}
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-title'>💬 Explanation & Advice</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-box'>{result['explanation']}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
        finally:
            os.unlink(temp_file_path)
