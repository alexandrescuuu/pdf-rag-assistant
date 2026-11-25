from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io


def build_demo_pdf() -> bytes:
    """Creates a multi-page technical AI Engineering handbook PDF in memory for instant testing."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    pages = [
        (
            "Page 1: Retrieval-Augmented Generation (RAG) Architecture",
            [
                "Retrieval-Augmented Generation reduces LLM hallucinations by grounding responses in external documents.",
                "Documents are split into overlapping chunks (typically 500 to 800 characters with 100-character overlap).",
                "Dense vector embeddings map semantic meaning, while sparse TF-IDF preserves exact keyword precision.",
                "Every generated claim must include an inline citation pointing to the exact source file and page number.",
            ],
        ),
        (
            "Page 2: Model Context Protocol & Tool-Using Agents",
            [
                "The Model Context Protocol (MCP) standardizes how AI assistants connect to local SQLite databases and APIs.",
                "Autonomous agents require circuit breakers that halt execution after 3 identical consecutive tool calls.",
                "OpenTelemetry instrumentation tracks input tokens, output tokens, USD cost, and span latency per step.",
            ],
        ),
        (
            "Page 3: LLM Security & Evaluation Pipelines",
            [
                "OWASP Top 10 for LLMs classifies Prompt Injection and Jailbreaks under risk category LLM01.",
                "Canary tokens embedded in system prompts detect unauthorized prompt leakage in production outputs.",
                "Automated CI/CD evaluation gates run LLM-as-a-Judge tests on every commit to block regressions.",
            ],
        ),
    ]

    for title, bullets in pages:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 740, title)
        c.setFont("Helvetica", 11)
        y = 700
        for line in bullets:
            c.drawString(60, y, f"• {line}")
            y -= 28
        c.showPage()

    c.save()
    return buf.getvalue()
