# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Legal RAG Ingestion — German federal law scraper, search index, and Q&A API.

Scrapes 72 German federal laws from gesetze-im-internet.de, processes them into structured paragraph-level documents, and indexes them for hybrid search (bge-m3 dense + learned sparse, weighted fusion, cross-encoder reranker). Deployed on Modal as a FastAPI web app with LEX/JURA legal assistant persona (DeepSeek default, Claude switchable).

**Current index**: 16,509 cleaned paragraph documents from 72 laws + 782 court ruling chunks from 398 Urteilen (BGH, BVerfG, BVerwG, BFH, BAG, BSG, BPatG).

## Quick commands

```bash
source .venv/bin/activate

# Scrape + index
python main.py --run-gesetze          # 72 German federal laws
python main.py --run-urteile          # 7 Bundesgerichte
python main.py --run-all              # both + EU law
python main.py --stats                # storage size + doc count

# Rebuild (filter out weggefallen/TOC junk, re-embed all)
python rebuild_clean.py

# Monitor rebuild progress
tail -f /tmp/claude-*/*/tasks/*.output
grep "Embedding:" /tmp/claude-*/*/tasks/*.output | tail -1

# Search
python main.py --search "Treu und Glauben" --top-k 5
python main.py --search "§ 242 BGB" --gesetz BGB
python main.py --search-related "BGB||§ 242"

# Deploy to Modal
modal deploy modal_deploy.py

# Upload rebuilt index to Modal Volume
modal volume put legal-rag-data documents.json /legal_rag_storage/documents.json
modal volume put legal-rag-data legal_graph.graphml /legal_rag_storage/legal_graph.graphml
modal volume put legal-rag-data qdrant/meta.json /legal_rag_storage/qdrant/meta.json
modal volume put legal-rag-data qdrant/collection/legal_docs/storage.sqlite /legal_rag_storage/qdrant/collection/legal_docs/storage.sqlite
```

## Architecture

```
gesetze-im-internet.de (XML)          rechtsprechung-im-internet.de (RSS + XML ZIP)
        │                                        │
        ▼                                        ▼
  GesetzeScraper ────► raw docs            UrteileScraper ────► raw docs
  (abkürzung, paragraph,                   (gericht, aktenzeichen, datum,
   paragraph_titel, inhalt, …)              leitsatz, volltext, norm, …)
        │                                        │
        └────────────────┬───────────────────────┘
                         ▼
        Processors (cleaner.py → metadata_extractor.py) ── only Urteile get chunked via chunker.py
                         │
                         ▼
               LegalRAGPipeline.insert_documents()
                         │
                         ├──► LegalEmbedder (FlagEmbedding BGEM3FlagModel, BAAI/bge-m3, 1024-dim dense + learned sparse)
                         │         │
                         │         ▼
                         │    Qdrant (local mode, SQLite, path=legal_rag_storage/qdrant/)
                         │    Named vectors: "" (dense 1024-dim) + "lexical" (sparse)
                         │
                         └──► Knowledge Graph (NetworkX DiGraph, regex §-Verweis parser, 1.23M edges)
                                   │
                                   ▼
                             legal_graph.graphml
```

All indexes rebuilt together from `documents.json`. No incremental updates — full rebuild only.

**Search flow**: Query → embed (dense + sparse) → two Qdrant queries (dense + sparse, top_k*3 each) → weighted fusion (0.7 dense / 0.3 sparse, min-max normalized) → candidates → FlagReranker cross-encoder → top_k results. §-reference pinning and law auto-filter applied before fusion.

## Storage layout

```
legal_rag_storage/          ← path from $LEGAL_RAG_STORAGE env var
├── documents.json          ← 17K+ docs, all structured fields (source of truth)
├── legal_graph.graphml     ← NetworkX DiGraph (1.23M edges)
└── qdrant/                 ← Qdrant local mode (~210 MB, 1024-dim dense + sparse)
```

Modal Volume `legal-rag-data` mirrors this at `/legal_rag_storage`.

**Important**: Volume upload paths are relative to Volume root. The Volume mounts at `/legal_rag_storage`, so remote path `documents.json` → `/legal_rag_storage/documents.json` in container. Do NOT prefix remote paths with `legal_rag_storage/`.

## Search engine (rag_pipeline.py)

**Classes**:
- `LegalEmbedder` — wraps FlagEmbedding `BGEM3FlagModel`, `.embed(texts)` → `(list[list[float]], list[SparseVector])`, `.embed_query(text)` → `(list[float], SparseVector)`. Manually L2-normalizes dense vectors (bge-m3 doesn't auto-normalize). `use_fp16=True` when CUDA available.
- `LegalIndexer` — manages Qdrant (dense + sparse named vectors) + NetworkX graph. `_ensure_collection()` creates collection with `sparse_vectors_config={"lexical": SparseVectorParams(...)}`. `index()` embeds in batches, upserts to Qdrant with dict vector `{"": dense, "lexical": sparse}`. No BM25.
- `LegalSearcher` — weighted fusion search + cross-encoder reranker. `search()` parses query for §/law → pins exact matches → runs two independent Qdrant queries (dense + sparse) → weighted fusion (0.7/0.3, min-max normalized) → calls `_rerank()` with lazy-loaded `FlagReranker("BAAI/bge-reranker-v2-m3")` → prepends pinned, returns `[:top_k]`.
- `LegalRAGPipeline` — orchestrator, owns embedder/indexer/searcher, provides `insert_documents()` and `search()`.

**Weighted Fusion**: Two independent Qdrant queries (dense + sparse "lexical", each `limit=top_k*3`), min-max normalized per result set, then merged with `DENSE_WEIGHT=0.7` / `SPARSE_WEIGHT=0.3`. Solves the sparse blind-spot problem where rare German legal compounds (e.g. "Sachmängel") got zero sparse weight under RRF. Docs only in one set get 0.0 for the other component.

**Reranker**: `FlagReranker("BAAI/bge-reranker-v2-m3")` lazy-loaded on first query. `compute_score(pairs, normalize=True)` — sigmoid-normalized scores in [0, 1]. Results gain `rerank_score` field, sorted descending. Only runs when `len(candidates) > 1`.

**§-reference pinning**: Queries with explicit paragraph references (regex `§+\s*\d+[a-z]?`) get exact matches pinned to top with score 1.0. Runs before weighted fusion to exclude pinned IDs from candidates.

**Law auto-filter**: Queries mentioning known law abbreviations automatically apply a Qdrant `gesetz` filter. Handles "GmbH" → "GmbHG" via `abk.rstrip("G")` fallback.

**Known abbreviations**: BGB, StGB, HGB, ZPO, StPO, GG, VwVfG, AktG, GmbH/GmbHG, InsO, FamFG, BDSG, UrhG, MarkenG, PatG, BauGB, VwGO, AO, KStG, EStG, UStG, UmwG, WpHG, BetrVG, SGB, KSchG, BVerfGG, TKG, WEG, EGBGB, BGBEG.

**Qdrant API**: Uses v1.17+ `query_points()` for independent dense + sparse queries. `Distance.COSINE` with named sparse vectors (`SparseIndexParams(full_scan_threshold=10000)`).

**Embedding speed** (CPU, no GPU locally): ~250 docs/min. Full 16.9K rebuild = ~60-70 min.

## rebuild_clean.py — Data quality filters

Loads `documents.json`, applies filters, saves cleaned version, then full rebuild:

| Filter | What it catches | Count removed |
|---|---|---|
| `inhalt` in ("-", "(weggefallen)", "(aufgehoben)") | Repealed paragraphs | 191 |
| `inhalt.startswith("-")` | Repeal notes with text after dash | 4 |
| `paragraph.startswith("BJNR")` or `"BJNG"` | XML node IDs (preambles, not real paragraphs) | 253 |
| `"Inhaltsübersicht"` or `"Inhaltsverzeichnis"` in paragraph | Table of contents entries | 35 |
| `abkürzung.startswith("./")` | Unofficial TOC (nichtamtliches Inhaltsverzeichnis) | 1,521 |

**Total**: ~17,500 raw → 17,291 clean (16,509 Gesetze + 782 Urteile).

## Modal deployment (modal_deploy.py)

**App**: `modal.App("legal-rag")`
**URL**: `https://aliundmaggy--legal-rag-fastapi-app.modal.run`

**Image**: Debian slim Python 3.12 with `FlagEmbedding>=1.2.0`, `transformers==4.57.6` (exact pin — newer transformers breaks `XLMRobertaTokenizer.prepare_for_model`). Pre-downloaded `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3` models. `src/` mounted via `add_local_dir(copy=True)`.

**GPU**: T4 (`gpu="T4"` on `@app.cls`). CPU embedding is ~7s/batch.

**Secrets**: `my-deepseek-secret` (DeepSeek API key), `my-anthropic-secret` (Anthropic API key).

**Classes**:
- `LegalRAG` (`@app.cls`, `@modal.concurrent(max_inputs=10)`, `gpu="T4"`) — loads embedder + index + reranker in `@modal.enter()`, exposes `.search()`, `.get_related()`, `.generate_answer()`, `.stats()` as `@modal.method()`.

**FastAPI routes**:
- `GET /api/legal/search?q=...&top_k=10&rechtsgebiet=...&gesetz=...`
- `GET /api/legal/ask?q=...&top_k=5` — search + LEX persona LLM answer with citations
- `GET /api/legal/ask/stream` — Server-Sent Events streaming
- `GET /api/legal/related/{doc_id}` — knowledge graph references
- `GET /api/legal/stats` — index statistics
- `GET /` — LEX chat UI (HTML/JS SPA with dark theme, German)

**Weekly cron**: `weekly_ingest` runs Saturday 02:00 UTC. **NOTE**: Modal IPs blocked by gesetze-im-internet.de, so cron scraping fails in production. Scrape locally, upload to Volume.

**LLM Provider Switch**: `LLM_PROVIDER` env var (`deepseek` default, `anthropic` for Claude). `generate_answer()` branches: DeepSeek via `openai.OpenAI(base_url="https://api.deepseek.com")` with `deepseek-chat`, Claude via `anthropic.Anthropic()` with `claude-sonnet-4-20250514`.

## Urteile Scraper (src/scrapers/urteile_scraper.py)

Scrapes court rulings from rechtsprechung-im-internet.de via RSS feeds + XML ZIP downloads. RSS feed (`/jportal/docs/feed/bsjrs-{court_slug}.xml`) → parse title/date/doc_id → download XML ZIP → extract structured fields.

**Courts**: BGH (201), BVerfG (29), BVerwG (54), BFH (56), BAG (29), BSG (24), BPatG (5).

**Constructor params**: `courts` (default `["BGH", "BVerfG"]`), `max_per_court`, `zivilsenat_only`.

**Output fields**: `typ="urteil"`, `gericht`, `senate`, `aktenzeichen`, `doktyp`, `ecli`, `datum`, `norm` (§ references), `vorinstanz`, `rechtsgebiet`, `leitsatz`, `volltext` (Leitsatz+Tenor+Tatbestand+Entscheidungsgründe), `url`, `quelle`, `doc_id`.

**Rate limiting**: 1 req/s.

## Key constraints

- **Scraper IP block**: gesetze-im-internet.de blocks Modal/datacenter IPs. All scraping must run locally.
- **Qdrant local mode**: Single-writer only. `ImportError` at exit is harmless (Python shutdown race).
- **FlagEmbedding version**: 1.4.0 local. Exact `transformers==4.57.6` pin on Modal.
- **Incremental indexing**: `insert_documents()` embeds only new docs. Partial scrapes append without re-embedding.
- **`documents.json` is the source of truth**: All rebuilds read from it.
- **`.env` file**: Local secrets only. Modal uses `my-deepseek-secret` / `my-anthropic-secret`.
- **No test suite**: pytest installed but tests are stubs.
