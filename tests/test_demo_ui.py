"""Static UI regressions for answer-quality critical API routing."""

from pathlib import Path


UI_PATH = Path(__file__).resolve().parents[1] / "src" / "static" / "demo_ui.html"


def test_chat_fallback_uses_enhanced_answer_pipeline_not_raw_search():
    html = UI_PATH.read_text(encoding="utf-8")
    assert "async function fallbackAsk" in html
    assert "/api/legal/ask/enhanced" in html
    assert "/api/legal/search?q=" not in html
    assert "Hybrid-Index" not in html
    assert "if (!answerText || !answerText.trim())" in html
    assert "fallbackAsk(queryWithCase);" in html


def test_streaming_ui_does_not_finalize_empty_placeholder_answers():
    html = UI_PATH.read_text(encoding="utf-8")
    assert "Erstelle juristisches Gutachten" not in html
    assert "updateStreamingMessage('')" not in html
    assert "if (!text || !text.trim()) {\n    if (msg) msg.remove();\n    return;\n  }" in html


def test_chat_labels_answers_as_checked_analysis():
    html = UI_PATH.read_text(encoding="utf-8")
    assert "LEX · geprüfte Analyse" in html


def test_chat_renders_and_persists_answer_audit_metadata():
    html = UI_PATH.read_text(encoding="utf-8")
    assert "answer_audit" in html
    assert "answerAudit" in html
    assert "renderAuditBadge" in html
    assert "renderAuditSection" in html
    assert "Quellenprüfung bestanden" in html
    assert "Anwaltliche Prüfung erforderlich" in html


def test_export_is_kanzlei_memo_with_audit_and_source_appendix():
    html = UI_PATH.read_text(encoding="utf-8")
    assert "# LEX — Kanzlei-Memo" in html
    assert "## Quellen- und Auditstatus" in html
    assert "## Anwalt-Handoff" in html
    assert "## Quellenanhang" in html
    assert "auditMarkdown" in html


def test_mobile_layout_has_compact_responsive_rules():
    html = UI_PATH.read_text(encoding="utf-8")
    assert "@media (max-width: 860px)" in html
    assert "--sidebar-w: min(86vw, 320px)" in html
    assert ".chat-header-actions button:not(.docs-btn)" in html
    assert ".audit-section" in html
    assert "env(safe-area-inset-bottom)" in html
