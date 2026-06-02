# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python Legal RAG system for German law-firm research workflows. The CLI is `main.py`; Modal deployment is in `modal_deploy.py`; cleanup/rebuild logic is in `rebuild_clean.py`.

- `src/scrapers/`: async scrapers for German federal laws, court rulings, and the EUR-Lex scaffold.
- `src/processors/`: text cleaning, metadata extraction, date normalization, and ruling chunking.
- `src/ingestion/`: Qdrant indexing, bge-m3 embeddings, dense/sparse fusion, reranking, and legal graph creation.
- `src/retrieval/`: query classification, rewriting, enhanced search, and deterministic legal quality controls.
- `src/retrieval/legal_quality.py`: source profiles, mandatory-source injection, false-positive filtering, `retrieval_plan`, and `source_audit`.
- `src/static/demo_ui.html`: Modal-served LEX chat UI.
- `tests/`: pytest coverage for ingestion, retrieval, quality gates, and UI routing regressions.

Do not commit generated indexes, logs, `.env*`, Modal tokens, API keys, or model artifacts.

## Current Runtime Status

Live app: `https://aliundmaggy--legal-rag-fastapi-app.modal.run`

Current Modal index:

- `17,024` indexed documents
- `74` German federal laws
- `16,242` law paragraphs
- `782` court-decision chunks
- `398` unique decisions
- `201` unique BGH decisions

Default development LLM provider is Gemini via `gemini-2.5-flash-lite`. DeepSeek and Anthropic are optional fallbacks.

## Build, Test, and Development Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Useful commands:

```bash
python main.py --stats
python main.py --search "§ 242 BGB" --gesetz BGB
python main.py --run-gesetze
python main.py --run-urteile
python rebuild_clean.py
modal deploy modal_deploy.py
```

Run tests:

```bash
python -m pytest -q
python -m pytest tests/test_legal_quality.py -q
python -m pytest tests/test_retrieval_quality.py -v -s
```

Latest fast-suite baseline: `52 passed, 3 skipped`.

## Coding Style & Naming Conventions

Use Python 3.12-compatible code, 4-space indentation, type hints on public pipeline boundaries, and concise module docstrings. Keep changes local to the relevant pipeline stage. Avoid broad refactors unless they directly improve retrieval quality, answer grounding, or deployment reliability.

Use `test_*.py` for tests. Prefer behavior-focused test names such as `test_chat_fallback_uses_enhanced_answer_pipeline_not_raw_search`.

## Legal Quality Rules

Do not rely on prompt wording alone for legal correctness. High-value legal issues should be encoded in `src/retrieval/legal_quality.py` as source profiles.

Each profile should define:

- trigger phrases and legal area
- required norms
- recommended norms
- excluded norms
- allowed law families
- answer-focus requirements

Every profile change needs regression tests proving:

- mandatory sources are injected
- known false positives are rejected
- unrelated legal areas do not enter the answer context
- `missing_required` behaves correctly when a source is absent

The UI must never fall back from chat to raw `/api/legal/search` output. Chat fallback must use `/api/legal/ask/enhanced`.

## Commit & Pull Request Guidelines

Use concise imperative commit subjects, for example:

- `Add deterministic legal quality layer`
- `Route chat fallback through enhanced analysis`
- `Refine ordinary termination source profile`

Pull requests should describe the changed pipeline stage, expected impact on retrieval/answer quality, storage or deployment impact, and verification commands. Include screenshots only for UI changes.

## Security & Configuration Tips

Keep secrets local and in Modal named secrets. Current Modal secrets:

- `my-gemini-secret`
- `my-deepseek-secret`
- `my-anthropic-secret`

Scraping must run from a local/residential network because some legal data sources block datacenter IPs. Respect the 1 request/second scrape rate. Never log API keys or user legal facts.

## Product Hardening Roadmap For Law Firms

Priority work for making the product maximally useful and defensible for Kanzleien:

1. **Expand Legal Quality Profiles**
   Add profiles for frequent law-firm workflows: employment dismissal, rent defects, rent termination, purchase defects, payment default, GmbH director liability, insolvency filing duty, fraud allegation, administrative objection, and limitation periods.

2. **Build A Kanzlei Eval Set**
   Maintain 100-150 representative legal questions with `must_include`, `must_not_include`, expected answer structure, risk level, and human-review notes. Treat regressions as blocking.

3. **Improve Citation Resolution**
   Replace or augment regex-only citation parsing with a robust German norm resolver. Evaluate `bundesrecht`-style parsing for normalized statute references and better `§§`, `Abs.`, `Satz`, and law-abbreviation handling.

4. **Separate Statute And Case-Law Retrieval**
   Retrieve statutes and decisions in separate channels, then fuse after validation. Answers should distinguish binding statutes, case-law support, and contextual commentary.

5. **Add Source-Level Answer Auditing**
   Add a post-generation verifier that checks every material claim against the provided context and flags unsupported claims, missing deadlines, missing required norms, and overconfident language.

6. **Broaden Legal Data Coverage**
   Evaluate Open Legal Data, additional BGH/BAG/BVerfG coverage, and official APIs where license-compatible. Track source provenance and update dates per document.

7. **Add Professional Workflow Features**
   Add matter context, document upload, structured intake, exportable memo, PDF/Word output, source appendix, and handoff notes for lawyers.

8. **Operational Reliability**
   Add API latency monitoring, Modal health checks, request tracing, cost tracking per provider, cached enhanced answers for repeated queries, and structured error reports.

9. **Compliance And Trust**
   Add clear disclaimers, data retention controls, tenant separation if multi-client, audit logs, and optional local-only storage for sensitive matter facts.
