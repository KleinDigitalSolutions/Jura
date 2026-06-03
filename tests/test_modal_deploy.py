import json
import math

import modal_deploy


def test_rag_container_serializes_shared_reranker_inputs():
    source = modal_deploy.Path(modal_deploy.__file__).read_text(encoding="utf-8")

    assert "@modal.concurrent(max_inputs=1)" in source
    assert "@modal.concurrent(max_inputs=10)" not in source


def test_async_web_handlers_use_modal_async_remote_calls():
    source = modal_deploy.Path(modal_deploy.__file__).read_text(encoding="utf-8")

    assert ".remote.aio(" in source
    assert ".remote(" not in source


def test_build_ask_context_coerces_nan_score_to_json_safe_float():
    _, citations = modal_deploy._build_ask_context(
        [
            {
                "abkürzung": "KSchG",
                "paragraph": "§ 4",
                "paragraph_titel": "Anrufung des Arbeitsgerichts",
                "inhalt": "Klage muss innerhalb von drei Wochen erhoben werden.",
                "score": math.nan,
            }
        ],
        "Kündigungsschutzklagefrist",
    )

    assert citations[0]["score"] == 0.0


def test_json_safe_replaces_non_finite_numbers_recursively():
    payload = modal_deploy._json_safe(
        {
            "score": math.nan,
            "results": [{"score": math.inf}, {"score": 0.75}],
        }
    )

    assert payload == {
        "score": None,
        "results": [{"score": None}, {"score": 0.75}],
    }
    json.dumps(payload, allow_nan=False)


def test_sse_serializes_strict_json_without_nan():
    event = modal_deploy._sse("search", {"citations": [{"score": math.nan}]})

    assert "event: search" in event
    assert "NaN" not in event
    assert '"score": null' in event
