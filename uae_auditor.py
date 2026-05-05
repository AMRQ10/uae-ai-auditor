"""Simple UAE AI Act 2026 high-risk keyword auditor.

This script checks AI system instructions for keywords that may indicate
classification as a "High-Risk System" under a simplified keyword method.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from typing import List, Optional, Set, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from groq import Groq
import streamlit as st
from fpdf import FPDF

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
EXEMPT_KEYWORDS: List[str] = [
    "scheduling",
    "formatting",
    "spellcheck",
]

DB_NAME = "audit_logs.db"

ANTHROPIC_AUDITOR_PROMPT = (
    "Act as a UAE AI Act Auditor. The following instructions triggered a high-risk keyword. "
    "Analyze the context and determine if this is a Tier 3/4 violation or an exempt administrative use-case. "
    'Return a JSON with "verdict" and "legal_reasoning".'
)

def run_ai_legal_analysis(text):
    """
    Uses Groq to perform a contextual risk assessment based on the UAE AI Act 2026.
    """
    # 1. Initialise the Groq client using my secret key
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        # 2. Craft the legal engineering import
        prompt = f"""
        You are a specialised Legal Engineer expert in the UAE AI Act 2026. 
        Analyze the following system instructions for a new AI Agent:

        "{text}"

        Task:
        1. Categorise the risk: Is this High-Risk' (Article 14), 'Prohibited', or 'Low-Risk'?
        2. Identify specific legal concerns: Look for biometric data behavioral manipulation, critical infrastructure.
        3. Provide a 3-sentence concise summary: (Risk level, Primary Legal Basis, and a Mitigation Recommendation).
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




def normalize_text(text: str) -> str:
    """Normalize whitespace and casing for consistent matching."""
    return " ".join(text.lower().split())


def find_matches(system_instructions: str, keywords: List[str]) -> List[str]:
    """Return sorted unique keywords found in system instructions."""
    normalized_instructions = normalize_text(system_instructions)
    found: Set[str] = set()

    for keyword in keywords:
        if normalize_text(keyword) in normalized_instructions:
            found.add(keyword)

    return sorted(found)


def get_classification(
    high_risk_matches: List[str], exempt_matches: List[str]
) -> tuple[str, str, str]:
    """Return classification label, color, and action guidance."""
    has_high_risk = len(high_risk_matches) > 0
    has_exempt = len(exempt_matches) > 0

    if has_high_risk and has_exempt:
        return (
            "WARNING (Contextual Review Needed)",
            "yellow",
            "Contextual Review Needed - assess intent, scope, and operational impact "
            "before final classification.",
        )
    if has_high_risk:
        return (
            "HIGH-RISK SYSTEM (Keyword Match Found)",
            "red",
            "Conduct full legal review and document risk controls.",
        )
    return (
        "NO HIGH-RISK KEYWORDS DETECTED",
        "green",
        "Keep monitoring; perform periodic reassessment.",
    )


def calculate_risk_score(
    high_risk_matches: List[str],
    exempt_matches: List[str],
    llm_audit: Optional[dict] = None,
) -> int:
    """
    Compute a 0–100 risk score:
    +10 per unique high-risk keyword match,
    +50 if Anthropic verdict is 'High-Risk',
    -20 if any exempt keyword is present.
    """
    score = 10 * len(high_risk_matches)

    if llm_audit:
        verdict = llm_audit.get("verdict")
        if verdict is not None:
            v = str(verdict).strip().lower().replace(" ", "-")
            if v == "high-risk":
                score += 50

    if exempt_matches:
        score -= 20

    return max(0, min(100, score))


def risk_score_theme(score: int) -> str:
    """Rich color name for score bands: 0–40 green, 41–70 yellow, 71–100 red."""
    if score <= 40:
        return "green"
    if score <= 70:
        return "yellow"
    return "red"


def format_risk_score_bar(score: int, *, total: int = 100, bar_width: int = 20) -> str:
    """ASCII bar plus fraction, e.g. [████████░░░░░░░░░░░░] 85/100."""
    filled = int(round((score / total) * bar_width)) if total else 0
    filled = max(0, min(bar_width, filled))
    bar = "█" * filled + "░" * (bar_width - filled)
    return f"[{bar}] {score}/{total}"


def _extract_json_object(text: str) -> str:
    """Strip optional markdown fences and return JSON object substring."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def anthropic_high_risk_review(
    system_instructions: str,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Call Anthropic when high-risk keywords matched. Returns (parsed_json, error_message).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            None,
            "ANTHROPIC_API_KEY is not set. Export it to enable AI legal reasoning.",
        )
    try:
        import anthropic  # type: ignore[import-untyped]
    except ImportError:
        return (
            None,
            "The 'anthropic' package is not installed. Run: pip install anthropic",
        )

    user_content = (
        f"{ANTHROPIC_AUDITOR_PROMPT}\n\n---\n\nSystem instructions:\n{system_instructions}"
    )
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as exc:
        return None, f"Anthropic API error: {exc}"

    if not message.content:
        return None, "Empty response from Anthropic API."

    block = message.content[0]
    raw = getattr(block, "text", None) or str(block)
    try:
        payload = json.loads(_extract_json_object(raw))
    except json.JSONDecodeError as exc:
        return None, f"Could not parse JSON from model response: {exc}"

    verdict = payload.get("verdict")
    reasoning = payload.get("legal_reasoning")
    if verdict is None and reasoning is None:
        return None, "Model JSON missing 'verdict' and 'legal_reasoning'."

    return {"verdict": verdict, "legal_reasoning": reasoning}, None


def init_db() -> None:
    """Create audit log table if it does not exist."""
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_text TEXT NOT NULL,
                flagged_keywords TEXT NOT NULL,
                risk_classification TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_scan_result(
    timestamp: str,
    input_text: str,
    flagged_keywords: List[str],
    risk_classification: str,
) -> None:
    """Persist one scan result into the SQLite audit log."""
    keyword_text = ", ".join(flagged_keywords) if flagged_keywords else "None"

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """
            INSERT INTO scan_logs (timestamp, input_text, flagged_keywords, risk_classification)
            VALUES (?, ?, ?, ?)
            """,
            (timestamp, input_text, keyword_text, risk_classification),
        )
        conn.commit()


def view_history() -> None:
    """Print a table of the last 5 scans from the SQLite audit log."""
    init_db()
    console = Console()

    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute(
            """
            SELECT timestamp, input_text, flagged_keywords, risk_classification
            FROM scan_logs
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()

    history_table = Table(
        title="Last 5 Compliance Scans",
        header_style="bold white",
        title_style="bold",
    )
    history_table.add_column("Timestamp", style="bold cyan")
    history_table.add_column("Classification", style="bold")
    history_table.add_column("Flagged Keywords", style="bold")
    history_table.add_column("Input Preview", style="dim")

    if not rows:
        history_table.add_row("-", "-", "-", "No scan history found")
    else:
        for timestamp, input_text, flagged_keywords, classification in rows:
            if "HIGH-RISK" in classification:
                classification_cell = f"[red]{classification}[/red]"
            elif "WARNING" in classification:
                classification_cell = f"[yellow]{classification}[/yellow]"
            else:
                classification_cell = f"[green]{classification}[/green]"

            preview = input_text.replace("\n", " ").strip()
            if len(preview) > 70:
                preview = preview[:67] + "..."

            history_table.add_row(
                timestamp,
                classification_cell,
                flagged_keywords,
                preview,
            )

    console.print(history_table)
    console.print()


def print_report(
    system_instructions: str,
    high_risk_matches: List[str],
    exempt_matches: List[str],
    *,
    llm_audit: Optional[dict] = None,
    llm_error: Optional[str] = None,
) -> None:
    """Print a professional compliance report in terminal using Rich."""
    console = Console()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    classification, _, action_needed = get_classification(
        high_risk_matches, exempt_matches
    )
    risk_score = calculate_risk_score(high_risk_matches, exempt_matches, llm_audit)
    theme = risk_score_theme(risk_score)
    risk_bar = format_risk_score_bar(risk_score)

    header = (
        f"[bold]Risk Score:[/bold] [{theme}]{risk_bar}[/{theme}]\n"
        f"[bold]Scan Timestamp:[/bold] {timestamp}\n"
        f"[bold]Input Length:[/bold] {len(system_instructions)} characters\n"
        f"[bold]Classification:[/bold] [{theme}]{classification}[/{theme}]"
    )
    console.print(
        Panel(
            header,
            title=f"[bold {theme}]UAE AI ACT 2026 - COMPLIANCE REPORT[/bold {theme}]",
            border_style=theme,
            expand=False,
        )
    )

    table = Table(
        title="Flagged Terms",
        header_style=f"bold {theme}",
        title_style=f"bold {theme}",
    )
    table.add_column("Type", style="bold")
    table.add_column("Keyword", style="bold")

    for keyword in high_risk_matches:
        table.add_row("[red]High-Risk[/red]", keyword)
    for keyword in exempt_matches:
        table.add_row("[yellow]Exempt Context[/yellow]", keyword)

    if not high_risk_matches and not exempt_matches:
        table.add_row("[green]None[/green]", "No flagged keywords found")

    console.print(table)

    if high_risk_matches:
        if llm_audit:
            verdict = llm_audit.get("verdict", "—")
            reasoning = llm_audit.get("legal_reasoning", "—")
            reasoning_str = str(reasoning) if reasoning is not None else "—"
            ai_body = (
                f"[bold]AI Verdict (Anthropic):[/bold] {verdict}\n\n"
                f"[bold]Legal Reasoning:[/bold]\n{reasoning_str}"
            )
            console.print(
                Panel(
                    ai_body,
                    title=f"[bold {theme}]UAE AI Act — LLM Contextual Review[/bold {theme}]",
                    border_style=theme,
                    expand=False,
                )
            )
        elif llm_error:
            console.print(
                Panel(
                    f"[yellow]{llm_error}[/yellow]",
                    title="[bold]LLM Review Unavailable[/bold]",
                    border_style="yellow",
                    expand=False,
                )
            )

    console.print(Text(f"Action Needed: {action_needed}", style=f"bold {theme}"))
    console.print()


def collect_system_instructions() -> str:
    """Collect multiline system instructions until user enters an empty line."""
    print("Enter AI System Instructions (press Enter twice to finish):")
    lines: List[str] = []

    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def choose_action() -> str:
    """Show an interactive startup menu and return selected action."""
    console = Console()
    menu = Table(title="UAE AI Auditor Menu", header_style="bold white", title_style="bold")
    menu.add_column("Option", style="bold cyan", width=10)
    menu.add_column("Action", style="bold")
    menu.add_row("1", "Run New Compliance Scan")
    menu.add_row("2", "View Last 5 Scans")
    menu.add_row("Q", "Quit")
    console.print(menu)

    while True:
        choice = input("Select an option (1/2/Q): ").strip().lower()
        if choice in {"1", "2", "q"}:
            return choice
        console.print("[yellow]Invalid choice. Please enter 1, 2, or Q.[/yellow]")


def main() -> None:
    """Run the compliance scanner."""
    if sys.version_info < (3, 9):
        print("Python 3.9+ is recommended for best compatibility.")

    if len(sys.argv) > 1 and sys.argv[1] == "--history":
        view_history()
        return

    action = choose_action()
    if action == "2":
        view_history()
        return
    if action == "q":
        print("Exiting UAE AI Auditor.")
        return

    instructions = collect_system_instructions()

    if not instructions:
        print("\nNo system instructions were provided. Exiting.")
        return

    init_db()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    high_risk_matches = find_matches(instructions, HIGH_RISK_KEYWORDS)
    exempt_matches = find_matches(instructions, EXEMPT_KEYWORDS)
    classification, _, _ = get_classification(high_risk_matches, exempt_matches)
    all_flagged_keywords = sorted(set(high_risk_matches + exempt_matches))
    save_scan_result(timestamp, instructions, all_flagged_keywords, classification)

    llm_audit: Optional[dict] = None
    llm_error: Optional[str] = None
    if high_risk_matches:
        llm_audit, llm_error = anthropic_high_risk_review(instructions)

    print_report(
        instructions,
        high_risk_matches,
        exempt_matches,
        llm_audit=llm_audit,
        llm_error=llm_error,
    )

def generate_pdf_report(score, ai_opinion, high_risk_found):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "UAE AI Act 2026 - Audit Certificate", ln= True, align="C")
    pdf.ln(10)

    # Risk Score Section
    pdf.set_font("Helvetica", "B", 12 )
    pdf.cell(0, 10, f"Calculated Risk Score: {score}/100", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 10, f"Flagged Keywords: {', '.join(high_risk_found) if high_risk_found else 'None'}")
    pdf.ln(5)

    # AI Opinion Selection
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Executive Legal Summary: ", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, ai_opinion)

    # Footer
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 10, "Generated by UAE AI Auditor 2026 - Internal Compliance Tool", align="C")

    return pdf.output # Returns byte string

if __name__ == "__main__":
    main()
