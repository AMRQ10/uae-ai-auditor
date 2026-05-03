import os

import streamlit as st

st.set_page_config(page_title="UAE AI Auditor 2026", page_icon="⚖️")

from uae_auditor import (
    EXEMPT_KEYWORDS,
    HIGH_RISK_KEYWORDS,
    calculate_risk_score,
    find_matches,
    run_ai_legal_analysis,
)

st.title("⚖️ UAE AI Act 2026 Auditor")
st.subheader("High-risk keyword scan (simplified web view)")

instructions = st.text_area("Enter system instructions here:", height=200)

if st.button("Run Compliance Audit"):
    if instructions.strip():
        high_risk = find_matches(instructions, HIGH_RISK_KEYWORDS)
        exempt = find_matches(instructions, EXEMPT_KEYWORDS)
        score = calculate_risk_score(high_risk, exempt, None)

        st.metric(label="Risk Score", value=f"{score}/100")

        with st.spinner("Groq is performing a deep legal analysis..."):
            try:
                ai_report = run_ai_legal_analysis(instructions)
                st.subheader("AI Legal opinion")
                st.info(ai_report)
            except Exception as e:
                st.error(f"AI analysis unavailable. Error: {e}")

        if score > 70:
            st.error("High risk detected (71–100).")
        elif score > 40:
            st.warning("Contextual review suggested (41–70).")
        else:
            st.success("Lower band (0–40). Continue monitoring.")

        if high_risk:
            st.write("**Flagged high-risk keywords**")
            for kw in high_risk:
                st.markdown(f"- {kw}")
        else:
            st.info("No high-risk list keywords matched.")

        if exempt:
            st.caption("Exempt-context keywords also found (score includes −20 adjustment):")
            for kw in exempt:
                st.markdown(f"- {kw}")
    else:
        st.info("Please enter instructions to audit.")

   

