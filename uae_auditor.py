"""Simple UAE AI Act 2026 high-risk keyword auditor.

This script checks AI system instructions for keywords that may indicate
classification as a "High-Risk System" under a simplified keyword method.
"""

import os
import re
import io
import json
from datetime import datetime
from typing import List, Optional
import streamlit as st
from fpdf import FPDF
from groq import Groq

# Simple example keyword list from UAE AI Act 2026 style categories.
HIGH_RISK_KEYWORDS: List[str] = [
    "biometric",
    "facial recognition",
    "emotion detection",
    "micro-expressions",
    "employee monitoring",
    "productivity tracking",
    "workplace surveillance",
    "critical infrastructure",
    "electricity grid",
    "water supply",
    "employment evaluation",
    "hiring automation",
    "cv screening",
    "law enforcement",
    "predictive policing",
    "parole assessment",
    "healthcare diagnosis",
    "surgical robotics",
    "patient triage",
    "education admissions",
    "exam proctoring",
    "student grading",
    "credit scoring",
    "loan eligibility",
    "insurance risk",
    "migration control",
    "visa processing",
    "border security",
    "public safety",
    "crowd management",
    "emergency response",
    "judicial decision",
    "legal precedent analysis",
    "case prediction"
]

# Terms that may indicate low-impact/admin use-cases and require context checks.
EXEMPT_KEYWORDS = [
    "scheduling",
    "formatting",
    "spellcheck"
]

def run_ai_legal_analysis(text):
    # 1. Initialise the Groq client using my secret key
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        # 2. Craft the legal engineering import
        prompt = f"""
        You are a specialised Legal Engineer expert in the UAE AI Act 2026. 
        Analyze the following system instructions for a new AI Agent:

        "{text}"

        Task:
        1. Categorise: High-Risk' (Article 14), 'Prohibited', or 'Low-Risk'?
        2. Identify legal concerns: biometric data, behavioral manipulation, critical infrastructure.
        3. A 3-sentence concise summary: (Risk level, Primary Legal Basis, and a Mitigation Recommendation).

        At the end of your report, you must provide a final numeric score on a new line in this exact format: FINAL_SCORE: X/100 where X is based on your legal assessment of the UAE AI Act.
        """

        #3. Request the completion form the Llama 3.3 model
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise UAE legal compliance expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, 
            max_tokens=500
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"AI Analysis Error: {str(e)}"


def find_matches(text: str, keywords: List[str]) -> List[str]:
    """Return sorted unique keywords found in system instructions."""
    normalized = text.lower()
    return sorted([kw for kw in keywords if kw.lower() in normalized])

def calculate_risk_score(high_risk_matches, exempt_matches, ai_report_text):
    """
    Hybrid Scoring: AI provides nuance, but Keywords provide a safety floor.
    """

    ai_score = 0

    match = re.search(r"FINAL_SCORE:\s*(\d+)", ai_report_text)

    if match:
        ai_score = int(match.group(1))

    floor = 70 if high_risk_matches else 0

    discount = len(exempt_matches) * 10 if ai_score < 90 else 0

    final_score = max(ai_score, floor) - discount

    return max(0, min(100, final_score))


def generate_pdf_report(score, ai_opinion, high_risk_found):
    pdf = FPDF()
    pdf.add_page()

    clean_opinion = ai_opinion.encode('latin-1', 'replace').decode('latin-1')

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "UAE AI Act 2026 - Audit Certificate", ln= True, align='C')
    pdf.ln(10)

    # Risk Score Section
    pdf.set_font("Helvetica", "B", 14 )
    pdf.cell(0, 10, f"Calculated Risk Score: {score}/100", ln=True)
    
    pdf.ln(5)

    # AI Opinion Selection
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Executive Legal Summary: ", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, ai_opinion)

    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M%S')}", align="C")

    pdf_bytes = pdf.output()
    if isinstance(pdf.bytes, str):
        pdf_bytes = pdf_bytes.encode('latin-1')
    
    return io.BytesIO(pdf_bytes)
