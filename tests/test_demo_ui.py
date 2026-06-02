"""Static UI regressions for answer-quality critical API routing."""

from pathlib import Path


UI_PATH = Path(__file__).resolve().parents[1] / "src" / "static" / "demo_ui.html"


def test_chat_fallback_uses_enhanced_answer_pipeline_not_raw_search():
    html = UI_PATH.read_text(encoding="utf-8")
    assert "async function fallbackAsk" in html
    assert "/api/legal/ask/enhanced" in html
    assert "/api/legal/search?q=" not in html
    assert "Hybrid-Index" not in html


def test_chat_labels_answers_as_checked_analysis():
    html = UI_PATH.read_text(encoding="utf-8")
    assert "LEX · geprüfte Analyse" in html
