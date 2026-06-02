# Repository Guidelines

## Project Structure & Module Organization

This is a Python Legal RAG ingestion and search project. The CLI is `main.py`; Modal deployment is in `modal_deploy.py`, and cleanup/rebuild logic is in `rebuild_clean.py`.

- `src/scrapers/`: async scrapers for German laws, court rulings, and EUR-Lex.
- `src/processors/`: text cleaning, metadata extraction, and document chunking.
- `src/ingestion/`: RAG indexing pipeline, Qdrant local mode, embeddings, reranking, and graph creation.
- `src/retrieval/`: query classification, query rewriting, and enhanced multi-query search.
- `src/static/`: local demo UI assets.
- `tests/`: pytest coverage for scrapers, processors, ingestion, and retrieval.
- `scripts/`: weekly scrape shell script and macOS launchd plist.
- `docker/`: container build and compose files.

Runtime storage is expected under `legal_rag_storage/` or an env-configured path. Do not commit generated indexes, logs, secrets, or model artifacts.

## Build, Test, and Development Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use `python main.py --stats` to inspect storage and `python main.py --search "§ 242 BGB" --gesetz BGB` for CLI search. Run `python main.py --run-gesetze` or `python main.py --run-urteile` to scrape and ingest. Rebuild with `python rebuild_clean.py`. Deploy with `modal deploy modal_deploy.py`.

Run tests with:

```bash
python -m pytest
python -m pytest tests/test_retrieval_quality.py -v -s
```

## Coding Style & Naming Conventions

Use Python 3.12-compatible code, 4-space indentation, and useful module docstrings. Prefer type hints for public functions and pipeline boundaries. Keep async scraper APIs consistent with `async with Scraper()`. Test files use `test_*.py`, test classes use `TestName`, and test methods use `test_behavior_expected_result`.

No formatter or linter configuration is checked in; keep changes close to existing style and avoid unrelated rewrites.

## Testing Guidelines

Use `pytest` and `pytest-asyncio` for async behavior. Mock network calls in unit tests; do not make live requests to legal data sources, Modal, or LLM providers. Add focused tests when changing scrapers, normalization, ID generation, indexing, or ranking.

## Commit & Pull Request Guidelines

Use concise imperative subjects such as `Add urteile chunk metadata` or `Fix sparse search fallback`. Pull requests should describe the changed pipeline stage, list verification commands, mention storage/index impact, and include screenshots only for UI changes.

## Security & Configuration Tips

Keep `.env` local. Required secrets include DeepSeek and optional Anthropic keys; Modal uses named secrets. Scraping must run from a local residential network because some sources block datacenter IPs. Respect the 1 request/second rate limit.
