<div align="center">

# Legal RAG — German Law Search & Q&A

**[Live Demo: legal-rag-fastapi-app.modal.run](https://aliundmaggy--legal-rag-fastapi-app.modal.run)**

**A production-oriented German legal retrieval and analysis system for law-firm workflows.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-ASGI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Modal](https://img.shields.io/badge/Modal-GPU_Deployment-7C3AED?style=for-the-badge)](https://modal.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid_Search-DC244C?style=for-the-badge)](https://qdrant.tech/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash_Lite-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Pytest](https://img.shields.io/badge/Tests-70_Passing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)

</div>

---

## Overview

**Legal RAG** is a hybrid retrieval-augmented generation platform for German law. It indexes German federal statutes and selected court decisions, combines dense and sparse semantic search with cross-encoder reranking, and generates citation-grounded legal analyses through a deterministic quality layer.

The goal is not a generic chatbot. The system is designed as a **law-firm research and first-analysis tool**: it classifies legal issues, retrieves mandatory legal sources, removes known false positives, and exposes auditable metadata such as `retrieval_plan`, `source_audit`, and post-generation `answer_audit`.

Current live index:

- **17,024** indexed documents
- **74** German federal laws
- **16,242** law paragraphs
- **782** court-decision chunks
- **398** unique court decisions
- **201** unique BGH decisions

> This project is a technical portfolio showcase. It is not a substitute for legal advice.

---

## Core Capabilities

### Legal Hybrid Search

- Dense semantic retrieval with `BAAI/bge-m3`
- Learned sparse vectors for exact legal terminology
- Weighted dense/sparse fusion with Qdrant
- Cross-encoder reranking via `BAAI/bge-reranker-v2-m3`
- Exact paragraph pinning for queries like `§ 242 BGB`
- Automatic law filtering for abbreviations such as `BGB`, `KSchG`, `StGB`, `InsO`

### German Legal Corpus

- Scrapes laws from [gesetze-im-internet.de](https://www.gesetze-im-internet.de)
- Scrapes court decisions from [rechtsprechung-im-internet.de](https://www.rechtsprechung-im-internet.de)
- Covers BGH, BVerfG, BVerwG, BFH, BAG, BSG, and BPatG
- Stores structured metadata: law abbreviation, paragraph, title, legal area, date, court, file number
- Builds a NetworkX knowledge graph from paragraph references

### Deterministic Legal Quality Layer

The most important production hardening is implemented in `src/retrieval/legal_quality.py`.

For recognized issue profiles, the system:

- injects mandatory legal sources before answer generation
- removes known false positives from the context
- restricts sources to allowed legal source families
- returns an auditable `source_audit`
- returns a structured `retrieval_plan`

Example: for ordinary employee termination, the system requires or recommends:

- `BGB § 623` — written form
- `BGB § 130` — receipt/access of declarations
- `BGB § 622` — notice periods
- `KSchG §§ 1, 4, 23` — dismissal protection, claim deadline, scope
- `BetrVG § 102` — works council hearing
- `SGB IX § 168` — disability-related approval requirement

It filters unrelated or misleading sources such as `BGB § 580a`, `BetrVG § 103`, `SGB IX § 175`, and `TzBfG § 16` for that profile.

### Source-Level Answer Auditing

Post-generation answer auditing is implemented in `src/retrieval/answer_audit.py` and returned from the enhanced answer endpoints as `answer_audit`.

The auditor is deterministic and dependency-free. It checks generated answers for:

- material legal claims without a source citation
- citation IDs that were not provided to the model
- paragraph or law references that do not match the cited sources
- required norms from the retrieval plan that are missing in the answer
- profile-specific deadline omissions, such as the KSchG § 4 three-week claim deadline
- overconfident wording such as "immer", "garantiert", or "zweifelsfrei"

The API returns an audit status (`pass`, `warn`, `fail`, or `error`), score, issue counts, and structured issue details. Prompt wording still asks the model to cite every material legal sentence, but the auditor is the enforcement layer.

### Kanzlei Eval Set

The first law-firm regression set lives in `evals/kanzlei_core.json`. It intentionally does **not** hardcode answers. Each case defines a quality contract:

- `must_include`: legal norms that must appear in the cited sources
- `must_not_include`: misleading or cross-domain norms that must not enter the answer context
- `expected_profiles`: deterministic retrieval profiles that should fire
- `max_high_audit_issues`: tolerated high-severity answer-audit issues
- `group`: `regression_guard` blocks regressions; `known_gap` tracks open product gaps without failing the gate

Run the eval gate against the enhanced answer API:

```bash
python scripts/run_kanzlei_eval.py --eval-set evals/kanzlei_core.json --skip-known-gaps
python scripts/run_kanzlei_eval.py --case arbeitsrecht_kuendigung_ordentlich_001 --json-report
```

This preserves the universal adviser pipeline: new facts still run through retrieval, quality profiles, generation, and answer auditing. The eval set only checks whether representative workflows keep their minimum source and audit guarantees.

### LEX Chat Interface

- FastAPI web UI served from Modal
- Server-Sent Events streaming endpoint
- Gemini development provider with DeepSeek and Claude fallback support
- Citation cards with source previews
- Dark law-firm-style interface
- Chat fallback always routes through the enhanced analysis pipeline, never raw search output

---

## Architecture

```text
gesetze-im-internet.de          rechtsprechung-im-internet.de
        |                                  |
        v                                  v
  GesetzeScraper                    UrteileScraper
        |                                  |
        +---------------+------------------+
                        v
        Cleaner -> MetadataExtractor -> Chunker
                        |
                        v
                 LegalRAGPipeline
                        |
        +---------------+------------------+
        |                                  |
        v                                  v
  bge-m3 embeddings                 NetworkX legal graph
 dense + learned sparse             paragraph references
        |
        v
 Qdrant local collection
        |
        v
 Weighted fusion -> Cross-encoder reranker
        |
        v
 EnhancedLegalSearch
 classify -> rewrite -> RRF -> quality audit
        |
        v
 FastAPI / Modal / LEX UI
```

| Layer | Technology |
|---|---|
| Runtime | Python 3.12 |
| API | FastAPI ASGI |
| Deployment | Modal, GPU T4, persistent Volume |
| Vector Search | Qdrant local mode |
| Embeddings | `BAAI/bge-m3` dense + sparse |
| Reranking | `BAAI/bge-reranker-v2-m3` |
| Legal Graph | NetworkX GraphML |
| LLM Providers | Gemini, DeepSeek, Anthropic |
| Tests | Pytest |

---

## API Endpoints

| Route | Purpose |
|---|---|
| `GET /` | LEX chat UI |
| `GET /api/legal/search` | Raw hybrid retrieval |
| `GET /api/legal/ask` | Search + generated answer |
| `GET /api/legal/ask/stream` | Streaming legal analysis |
| `GET /api/legal/ask/enhanced` | Enhanced retrieval + source/answer audits + generated answer |
| `GET /api/legal/related/{doc_id}` | Knowledge-graph related paragraphs |
| `GET /api/legal/stats` | Index statistics |

Live endpoint:

```bash
curl --get "https://aliundmaggy--legal-rag-fastapi-app.modal.run/api/legal/ask/enhanced" \
  --data-urlencode "q=Welche Anforderungen gelten für eine ordentliche Kündigung eines Arbeitnehmers?" \
  --data-urlencode "top_k=8"
```

---

## Project Structure

```text
.
├── main.py                         # CLI entry point
├── modal_deploy.py                 # Modal FastAPI deployment + LEX persona
├── rebuild_clean.py                # index cleanup and rebuild logic
├── evals/
│   └── kanzlei_core.json           # source/audit regression contracts
├── src/
│   ├── scrapers/                   # German laws, rulings, EUR-Lex scaffold
│   ├── processors/                 # cleaning, metadata, chunking
│   ├── ingestion/rag_pipeline.py   # embeddings, Qdrant, fusion, reranking
│   ├── retrieval/
│   │   ├── enhanced_search.py      # classify -> rewrite -> RRF -> quality layer
│   │   ├── answer_audit.py         # post-generation source-level answer audit
│   │   ├── legal_quality.py        # deterministic source profiles and audits
│   │   ├── query_classifier.py
│   │   ├── eval_runner.py          # deterministic eval-set checks
│   │   └── query_rewriter.py
│   ├── scheduler/
│   └── static/demo_ui.html
├── tests/
│   ├── test_legal_quality.py
│   ├── test_answer_audit.py
│   ├── test_eval_runner.py
│   ├── test_demo_ui.py
│   └── test_retrieval_quality.py
├── scripts/
└── requirements.txt
```

Runtime storage is expected under `legal_rag_storage/` or `$LEGAL_RAG_STORAGE`.

---

## Local Development

### Prerequisites

- Python 3.12
- Modal account for deployment
- Gemini API key for development answers
- Local/residential network for scraping German legal sources

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

```env
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash-lite
LLM_PROVIDER=gemini

DEEPSEEK_API_KEY=...
ANTHROPIC_API_KEY=...

LEGAL_RAG_STORAGE=legal_rag_storage
EMBEDDING_MODEL_NAME=BAAI/bge-m3
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
```

### CLI Usage

```bash
python main.py --stats
python main.py --search "Treu und Glauben" --top-k 5
python main.py --search "§ 242 BGB" --gesetz BGB
python main.py --run-gesetze
python main.py --run-urteile
```

---

## Deployment

```bash
modal setup
modal deploy modal_deploy.py
```

Upload a rebuilt index to the Modal Volume:

```bash
modal volume put legal-rag-data documents.json legal_rag_storage/documents.json
modal volume put legal-rag-data legal_graph.graphml legal_rag_storage/legal_graph.graphml
modal volume put legal-rag-data qdrant/meta.json legal_rag_storage/qdrant/meta.json
modal volume put legal-rag-data qdrant/collection/legal_docs/storage.sqlite legal_rag_storage/qdrant/collection/legal_docs/storage.sqlite
```

Modal configuration:

- App: `legal-rag`
- Volume: `legal-rag-data`
- Secrets: `my-gemini-secret`, `my-deepseek-secret`, `my-anthropic-secret`
- GPU: T4
- Live URL: [https://aliundmaggy--legal-rag-fastapi-app.modal.run](https://aliundmaggy--legal-rag-fastapi-app.modal.run)

---

## Testing & Validation

```bash
python -m pytest -q
python -m pytest tests/test_legal_quality.py -q
python -m pytest tests/test_answer_audit.py tests/test_eval_runner.py -q
python -m pytest tests/test_retrieval_quality.py -v -s
```

Current fast suite:

```text
70 passed, 3 skipped
```

Quality checks include:

- KG expansion regressions
- legal source filtering
- post-generation answer grounding audit
- Kanzlei eval-set schema and regression-gate checks
- mandatory-source injection
- UI routing guards against raw-search fallback
- retrieval quality evaluation across legal domains

---

## Known Limitations

- Index currently covers selected German federal law and selected court decisions, not every possible legal source.
- EUR-Lex integration is scaffolded but not yet production-integrated.
- No historical law versioning yet.
- Scraping must run from a residential/local network because some sources block datacenter IPs.
- Legal quality profiles are being expanded iteratively by high-value legal issue type.

---

## License

**Proprietary / Portfolio Showcase**  
All rights reserved by Klein Digital Solutions.

<div align="center">
  <sub>Built as a German legal AI research and law-firm automation showcase.</sub>
</div>
