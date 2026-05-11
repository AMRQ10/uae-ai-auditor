import os

import streamlit as st

st.set_page_config(page_title="UAE AI Auditor 2026", page_icon="⚖️")

from uae_auditor import (
    EXEMPT_KEYWORDS,
    HIGH_RISK_KEYWORDS,
    calculate_risk_score,
    find_matches,
    run_ai_legal_analysis,
    generate_pdf_report,
)

st.title("⚖️ UAE AI Act 2026 Auditor")
st.subheader("High-risk keyword scan (simplified web view)")

instructions = st.text_area("Enter system instructions here:", height=200)

if st.button("Run Compliance Audit"):
    if instructions.strip():
        high_risk = find_matches(instructions, HIGH_RISK_KEYWORDS)
        exempt = find_matches(instructions, EXEMPT_KEYWORDS)

        with st.spinner("Groq is performing a deep legal analysis..."):
            try:
                ai_report = run_ai_legal_analysis(instructions)

                score = calculate_risk_score(high_risk, exempt, ai_report)

                st.metric(label="Risk Score", value=f"{score}/100")
                st.subheader("AI Legal opinion")
                st.info(ai_report)

                if score >= 75:
                    st.error("HIGH RISK / PROHIBITED: Significant legal concerns detected.")
                elif score >= 40:
                    st.warning("CAUTION: Contextual review required.")
                else:
                    st.success("LOW RISK: No significant violations detected.")
                
                if high_risk:
                    st.write("**Flagged high-risk keywords**")
                    for kw in high_risk:
                        st.markdown(f"- {kw}")

                if exempt:
                    st.caption ("Exempt-context keywords also found (score adjustment applied):")
                    for kw in exempt:
                        st.markdown(f"- {kw}")

                st.markdown("---")
                report_buffer = generate_pdf_report(score, ai_report, high_risk)
                st.download_button(
                    label="Download Audit Report (PDF)",
                    data=report_buffer.getvalue(),
                    file_name="UAE_AI_Audit_Report.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"AI analysis unavailable. Error: {e}")
    else:
        st.info("Please enter instructions to audit.")
   

