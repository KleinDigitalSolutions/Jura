# Legal RAG — German Law Search & Q&A

Hybrid search engine for German federal law: 16,509 paragraphs from 72 laws + 782 court ruling chunks from 398 Urteilen (BGH, BVerfG, BVerwG, BFH, BAG, BSG, BPatG) indexed with bge-m3 (dense + learned sparse) + cross-encoder reranker. Deployed on Modal as a FastAPI web app with LEX/JURA legal assistant persona (DeepSeek default, Claude switchable).

**Live**: [legal-rag-fastapi-app.modal.run](https://aliundmaggy--legal-rag-fastapi-app.modal.run)

## Architecture

```
gesetze-im-internet.de (XML ZIPs)    rechtsprechung-im-internet.de (RSS + XML)
        │                                        │
        ▼                                        ▼
  GesetzeScraper ────────► raw docs       UrteileScraper ────────► raw docs
        │                                        │
        └────────────────┬───────────────────────┘
                         ▼
               Processors (clean → metadata → chunk Urteile)
                         │
                         ▼
               LegalRAGPipeline
                         │
                         ├──► LegalEmbedder (bge-m3) ──► Qdrant (1024-dim dense + learned sparse)
                         │                                      │
                         │                                      ▼
                         │                              Weighted Fusion (0.7 dense / 0.3 sparse)
                         │                                      │
                         │                                      ▼
                         │                              FlagReranker (bge-reranker-v2-m3)
                         │
                         └──► NetworkX DiGraph ──► legal_graph.graphml (§ references)
```

**No LLM during indexing** — regex §-parser for knowledge graph edges. Full rebuild: ~65 min CPU (12,766 docs), ~15 min GPU.

| Component | Technology |
|-----------|-----------|
| Embeddings | `BAAI/bge-m3` via FlagEmbedding (1024-dim dense + learned sparse) |
| Vector DB | Qdrant local mode (SQLite), named sparse vectors |
| Fusion | Weighted (0.7 dense / 0.3 sparse, min-max normalized) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder via FlagEmbedding |
| Knowledge Graph | NetworkX DiGraph, regex §-Verweis parser (341K edges) |
| LLM (Q&A) | DeepSeek Chat (default) / Claude Sonnet 4.6 (switchable via `LLM_PROVIDER`) |
| Deployment | Modal (FastAPI ASGI, Volume, GPU T4, Cron) |

## Quick Start

```bash
cd /Users/bucci369/legal-rag-ingestion
source .venv/bin/activate

# Scrape + index German federal laws (~2 min scrape + ~58 min index on CPU)
python main.py --run-gesetze

# CLI search
python main.py --search "Treu und Glauben" --top-k 5
python main.py --search "§ 242 BGB" --gesetz BGB

# Show stats
python main.py --stats
```

## Search Features

**Weighted Fusion** — Dense (1024-dim) + learned sparse vectors fused with 0.7/0.3 weights, min-max normalized per result set. Handles rare German legal compounds that sparse vectors alone miss.

**Cross-encoder Reranker** — `bge-reranker-v2-m3` jointly scores (query, document) pairs. Takes top-k*3 weighted fusion candidates, reranks, returns top-k. Results include both `score` (weighted fusion) and `rerank_score` (cross-encoder).

**§-reference pinning** — Queries containing explicit paragraph references (`§ 242`, `§ 823 BGB`) pin the exact match to #1 with score 1.0.

**Law auto-filter** — Queries mentioning known law abbreviations (BGB, StGB, GmbH, HGB, ZPO, etc.) automatically restrict vector search to that law's paragraphs. "GmbH Geschäftsführer Pflichten" → only GmbHG paragraphs searched.

## Data Pipeline

### Scraping
- **Gesetze**: [gesetze-im-internet.de](https://www.gesetze-im-internet.de)
- **Method**: Parse Teilliste index → download XML ZIP per law → extract paragraphs from `metadaten/enbez` + text from `textdaten/text/P`
- **Rate limiting**: 1 req/s, retry with exponential backoff (max 3)
- **72 laws**: GG, BGB, StGB, HGB, ZPO, GmbHG, AktG, VwVfG, InsO, SGB I–XII, and more
- **Urteile**: [rechtsprechung-im-internet.de](https://www.rechtsprechung-im-internet.de)
- **Method**: RSS feed → XML ZIP per ruling → extract Leitsatz, Tenor, Tatbestand, Entscheidungsgründe
- **Courts**: BGH (201), BVerfG (29), BVerwG (54), BFH (56), BAG (29), BSG (24), BPatG (5)
- **Priority**: BGH Zivilsenat > BGH Strafsenat > BVerfG
- **Chunking**: Urteile only — Leitsatz as own chunk, then sliding window (1000/200 token) on Tatbestand/Entscheidungsgründe

### Processing
1. **Clean** — HTML→text via BeautifulSoup+lxml, normalize § symbols
2. **Metadata** — extract legal area (10 categories), paragraph references, normalize dates
3. **Chunk** — Gesetze: 1 paragraph = 1 doc (no chunking). Urteile: Leitsatz + sliding window (1000/200 token) on Tatbestand + Entscheidungsgründe sections

### Indexing
All indexes rebuilt together from `documents.json`. No incremental updates.

### Data Quality Filters
Applied by `rebuild_clean.py`:

| Filter | Removes |
|--------|---------|
| `inhalt` in ("-", "(weggefallen)", "(aufgehoben)") | Repealed placeholder paragraphs |
| `inhalt.startswith("-")` | Repeal notes with text after dash |
| `paragraph` starts with BJNR/BJNG | XML node IDs (preambles) |
| `paragraph` contains "Inhaltsübersicht" | Table of contents entries |
| `abkürzung` starts with "./" | Unofficial TOC |

**17,500 raw → 17,291 clean documents (16,509 Gesetze + 782 Urteile).**

## Storage

```
legal_rag_storage/
├── documents.json          ← 17,291 docs, all structured fields
├── legal_graph.graphml     ← NetworkX DiGraph (1.23M edges)
└── qdrant/                 ← Qdrant local mode (~250 MB, 1024-dim + sparse)
```

Path set via `$LEGAL_RAG_STORAGE` env var.

## Modal Deployment

```bash
# Deploy
modal deploy modal_deploy.py

# Upload rebuilt index after local scrape (remote path relative to Volume root)
modal volume put legal-rag-data documents.json legal_rag_storage/documents.json
modal volume put legal-rag-data legal_graph.graphml legal_rag_storage/legal_graph.graphml
modal volume put legal-rag-data qdrant/meta.json legal_rag_storage/qdrant/meta.json
modal volume put legal-rag-data qdrant/collection/legal_docs/storage.sqlite legal_rag_storage/qdrant/collection/legal_docs/storage.sqlite
```

### API Endpoints

| Route | Description |
|-------|-------------|
| `GET /api/legal/search?q=...&top_k=10&rechtsgebiet=...&gesetz=...` | Weighted fusion search + reranker |
| `GET /api/legal/ask?q=...&top_k=5` | Search + LEX persona LLM answer with citations |
| `GET /api/legal/related/{doc_id}` | Knowledge graph references (bidirectional) |
| `GET /api/legal/stats` | Index statistics |
| `GET /` | LEX chat UI (dark theme, mobile-responsive) |

### Infrastructure
- **Image**: Debian slim Python 3.12 + pre-downloaded bge-m3 + reranker models
- **GPU**: T4 (recommended for bge-m3 + reranker inference)
- **Volume**: `legal-rag-data` at `/legal_rag_storage` (persistent)
- **Secrets**: `my-deepseek-secret` (DeepSeek API key), `my-anthropic-secret` (Anthropic API key, optional)
- **Concurrency**: max 10 inputs per container
- **Cron**: Saturday 02:00 UTC re-scrape (**broken** — Modal IPs blocked by gesetze-im-internet.de)

### LLM Provider Switch

Set `LLM_PROVIDER` env var to switch between DeepSeek (default) and Claude:

```bash
# DeepSeek (default, no config needed)
LLM_PROVIDER=deepseek

# Claude Sonnet 4.6 (requires my-anthropic-secret)
LLM_PROVIDER=anthropic
```

On Modal, set via Secret or environment variable in `modal_deploy.py`.

## CLI Reference

```bash
python main.py --run-all          # All scrapers + ingestion
python main.py --run-gesetze      # German federal laws only
python main.py --run-eurlex       # EU law only (scaffolded)
python main.py --run-urteile      # Court rulings only (BGH + BVerfG, ~230 rulings)
python main.py --schedule         # Start weekly scheduler
python main.py --stats            # Show storage size + doc/graph/Qdrant counts
python main.py --search QUERY     # CLI search (weighted fusion + reranker)
python main.py --search-related "BGB||§ 242"  # KG references
```

## Configuration (.env)

```bash
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...      # optional, for Claude switch
LLM_PROVIDER=deepseek             # deepseek or anthropic
LEGAL_RAG_STORAGE=legal_rag_storage
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DIM=1024
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
BATCH_SIZE=32
LOG_LEVEL=INFO
```

## Known Limitations

- **GPU recommended**: bge-m3 (2.2GB) + reranker (1.1GB) run on CPU but ~7s/batch. Modal T4 GPU reduces cold start + inference time significantly.
- **Court rulings**: 7 courts (BGH, BVerfG, BVerwG, BFH, BAG, BSG, BPatG) from rechtsprechung-im-internet.de (RSS feed + XML ZIP). BGH Zivilsenat prioritized. 398 rulings, chunked into 782 searchable segments. Scraped locally, uploaded to Volume.
- **No EU law**: EUR-Lex SPARQL scraper is scaffolded but not yet integrated.
- **Modal scraping broken**: gesetze-im-internet.de blocks datacenter IPs. All scraping must run locally, then upload to Volume.
- **No law versioning**: Indexes only the current version. No historical law versions or transitional provisions.
- **Single-writer Qdrant**: Local mode doesn't support concurrent writes. Harmless `sys.meta_path is None` error on process exit.
- **Partial scrapers merge**: `--run-urteile` or `--run-gesetze` alone no longer overwrite `documents.json`. Pipeline merges with existing docs (Gesetze dedup by `ABK||§`, Urteile by `doc_id`).

## Project Structure

```
legal-rag-ingestion/
├── src/
│   ├── scrapers/{base,gesetze,eurlex,urteile}_scraper.py
│   ├── processors/{cleaner,metadata_extractor,chunker}.py
│   ├── ingestion/rag_pipeline.py     ← core: bge-m3 embed, weighted fusion, rerank, search
│   ├── scheduler/cron.py
│   └── config.py
├── modal_deploy.py                   ← Modal FastAPI app + LEX persona
├── rebuild_clean.py                  ← filter + rebuild index from documents.json
├── main.py                           ← CLI entry point
├── tests/
├── .env.example
└── requirements.txt
```
