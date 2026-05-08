"""Modal deployment for Legal RAG — hybrid search + LLM answer generation.

Deploy:  modal deploy modal_deploy.py
Test:    curl https://aliundmaggy--legal-rag-fastapi-app.modal.run/api/legal/search?q=Treu+und+Glauben
"""

import json
import os
import sys
import time
from pathlib import Path

import modal
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------
app = modal.App("legal-rag")

# Persistent storage for Qdrant vectors, knowledge graph, documents
VOLUME = modal.Volume.from_name("legal-rag-data", create_if_missing=True)
VOLUME_PATH = Path("/legal_rag_storage")

# Secrets
DEEPSEEK_SECRET = modal.Secret.from_name("my-deepseek-secret")
ANTHROPIC_SECRET = modal.Secret.from_name("my-anthropic-secret")

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.115.6",
        "FlagEmbedding>=1.2.0",
        "transformers==4.57.6",
        "qdrant-client>=1.13.0",
        "networkx>=3.0",
        "loguru>=0.7.0",
        "openai>=1.0.0",          # DeepSeek API (OpenAI-compatible)
        "anthropic>=0.30.0",      # Claude API
        "aiohttp>=3.9.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=5.0.0",
        "python-dotenv>=1.0.0",
        "tenacity>=8.0.0",
        "scikit-learn>=1.0.0",
    )
    # Pre-download models into image (speeds up cold start)
    .run_commands([
        "echo 'build: weighted-fusion-v2'",
        f"python -c \"from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('{EMBEDDING_MODEL}', use_fp16=False)\"",
        f"python -c \"from FlagEmbedding import FlagReranker; FlagReranker('{RERANKER_MODEL}', use_fp16=False)\"",
    ])
    # src/ copied into image (copy=True to avoid stale cache on redeploy)
    .add_local_dir(str(Path(__file__).parent / "src"), remote_path="/src", copy=True)
)

# ---------------------------------------------------------------------------
# LEX/JURA System Prompt
# ---------------------------------------------------------------------------
LEX_SYSTEM_PROMPT = """Du bist LEX, ein KI-Rechtsassistent einer deutschen Anwaltskanzlei.
Du führst das Erstgespräch mit Mandanten — freundlich, professionell, strukturiert.

═══════════════════════════════════
DEINE ROLLE
═══════════════════════════════════

Du bist der erste Kontakt des Mandanten mit der Kanzlei.
Dein Ziel: Die Frage des Nutzers auf Basis der zitierten Gesetzestexte
präzise beantworten, sodass der Anwalt optimal vorbereitet ist.

Du bist KEIN Anwalt. Du gibst KEINE Rechtsberatung.
Du bist ein hochintelligenter juristischer Assistent,
der Informationen strukturiert aufnimmt und erklärt.

═══════════════════════════════════
REGELN FÜR DEINE ANTWORT
═══════════════════════════════════

1. ZITIERE JEDE QUELLE mit ihrer [Nummer] aus dem Kontext.
   Keine Aussage ohne Quellenangabe.

2. FALLS die Gesetzestexte die Frage nicht beantworten können,
   sage das ehrlich. Falls sie nur teilweise helfen, erkläre was fehlt.

3. STRUKTURIERE deine Antwort:
   - Relevante Gesetzesgrundlage nennen
   - Konkrete Paragraphen zitieren (mit [Nummer])
   - Nächste logische Schritte skizzieren

4. IMMER mit Disclaimer: "Dies ist eine allgemeine Information,
   keine Rechtsberatung. Ein Anwalt wird Ihren konkreten Fall bewerten."

5. Fasse am Ende die Kernaussage in 1-2 Sätzen zusammen.

═══════════════════════════════════
TONALITÄT
═══════════════════════════════════

- Ruhig, empathisch, professionell
- Kurze Sätze, klare Struktur
- Siezen immer
- Keine Fremdwörter ohne Erklärung

═══════════════════════════════════
ABSOLUT VERBOTEN
═══════════════════════════════════

- "Sie werden den Fall gewinnen"
- Konkrete Kostenaussagen
- Aussagen ohne RAG-Grundlage
- Politische oder moralische Bewertungen"""


# ---------------------------------------------------------------------------
# Load index once per container via @app.cls + @modal.enter()
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    volumes={VOLUME_PATH: VOLUME},
    secrets=[DEEPSEEK_SECRET, ANTHROPIC_SECRET],
    gpu="T4",
)
@modal.concurrent(max_inputs=10)
class LegalRAG:
    @modal.enter()
    def load(self):
        """Load embedding model + full index from Volume on container start."""
        sys.path.insert(0, "/")  # src/ is mounted at /src/

        os.environ["LEGAL_RAG_STORAGE"] = str(VOLUME_PATH)

        from src.ingestion.rag_pipeline import LegalEmbedder, LegalIndexer, LegalSearcher

        t0 = time.monotonic()
        self.embedder = LegalEmbedder(model_name=EMBEDDING_MODEL)
        self.indexer = LegalIndexer(embedder=self.embedder)
        self.searcher = LegalSearcher(self.indexer)
        if not self.indexer.load():
            print("WARNING: No index found on Volume — run ingest first")
        self.total_docs = len(self.indexer.documents)
        print(f"LegalRAG loaded: {self.total_docs} docs in {time.monotonic() - t0:.1f}s")

    @modal.method()
    def search(self, query: str, top_k: int = 10, rechtsgebiet: str = None, gesetz: str = None) -> list[dict]:
        return self.searcher.search(query, top_k=top_k, rechtsgebiet=rechtsgebiet, gesetz=gesetz)

    @modal.method()
    def get_related(self, doc_id: str) -> list[dict]:
        doc = self.indexer._para_index.get(doc_id)
        if not doc:
            return []
        return self.searcher.get_related(doc)

    @modal.method()
    def stats(self) -> dict:
        """Return index statistics."""
        return {"total_docs": len(self.indexer.documents)}

    @modal.method()
    def generate_answer(self, query: str, top_k: int = 5) -> dict:
        """Search + LLM answer with citations. Provider-switchable (deepseek/anthropic)."""
        search_results = self.searcher.search(query, top_k=top_k)

        if not search_results:
            return {"answer": "Keine relevanten Gesetzesstellen gefunden.", "citations": []}

        user_prompt, citations = _build_ask_context(search_results, query, top_k)

        provider = os.getenv("LLM_PROVIDER", "deepseek")

        if provider == "anthropic":
            try:
                from anthropic import Anthropic

                client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    temperature=0.2,
                    system=LEX_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                answer = response.content[0].text.strip()
                model_used = "claude-sonnet-4-20250514"
            except Exception as e:
                answer = f"Fehler bei der Anthropic-LLM-Antwortgenerierung: {e}"
                model_used = "error"
        else:
            try:
                from openai import OpenAI

                client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                    base_url="https://api.deepseek.com",
                )
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": LEX_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                answer = response.choices[0].message.content.strip()
                model_used = "deepseek-chat"
            except Exception as e:
                answer = f"Fehler bei der DeepSeek-LLM-Antwortgenerierung: {e}"
                model_used = "error"

        return {
            "answer": answer,
            "citations": citations,
            "model": model_used,
        }


# ---------------------------------------------------------------------------
# Shared helpers for answer generation
# ---------------------------------------------------------------------------
def _build_ask_context(search_results: list, query: str, top_k: int = 5):
    """Build LLM prompt and citations from search results."""
    context_parts: list[str] = []
    citations: list[dict] = []
    for i, r in enumerate(search_results[:top_k], 1):
        cite_id = f"[{i}]"
        abk = r.get("abkürzung", "")
        para = r.get("paragraph", "")
        titel = r.get("paragraph_titel", "")
        inhalt = r.get("inhalt", "") or r.get("volltext", "") or ""
        if len(inhalt) > 2000:
            inhalt = inhalt[:2000] + "…"
        context_parts.append(f"{cite_id} {para} {abk} — {titel}\n{inhalt}")
        citations.append({
            "id": cite_id,
            "gesetz": abk,
            "paragraph": para,
            "titel": titel,
            "url": r.get("url", ""),
            "score": r.get("rerank_score", r.get("score", 0)),
            "text_preview": inhalt[:2000],
        })
    context = "\n\n---\n\n".join(context_parts)
    user_prompt = f"""FRAGE: {query}

GESETZESTEXTE:
{context}

Beantworte die Frage auf Basis der zitierten Gesetzestexte.
Zitiere jede Quelle mit ihrer [Nummer]."""
    return user_prompt, citations


# ---------------------------------------------------------------------------
# FastAPI web app
# ---------------------------------------------------------------------------
web_app = FastAPI(title="Legal RAG API")


@web_app.get("/api/legal/search")
async def legal_search(q: str = "", top_k: int = 10, rechtsgebiet: str = "", gesetz: str = ""):
    """Hybrid search endpoint."""
    if not q:
        return {"error": "Query parameter 'q' required"}
    rag = LegalRAG()
    results = rag.search.remote(
        q, top_k=top_k,
        rechtsgebiet=rechtsgebiet or None,
        gesetz=gesetz or None,
    )
    return {"query": q, "results": results, "count": len(results)}


@web_app.get("/api/legal/ask")
async def legal_ask(q: str = "", top_k: int = 5):
    """Search + LLM answer generation."""
    if not q:
        return {"error": "Query parameter 'q' required"}
    rag = LegalRAG()
    result = rag.generate_answer.remote(q, top_k=top_k)
    return result


@web_app.get("/api/legal/ask/stream")
async def legal_ask_stream(q: str = "", top_k: int = 5):
    """Search + streaming LLM answer via Server-Sent Events."""
    if not q:
        return {"error": "Query parameter 'q' required"}
    rag = LegalRAG()

    async def event_stream():
        try:
            search_results = rag.search.remote(q, top_k=top_k)

            if not search_results:
                yield "event: error\ndata: " + json.dumps({"message": "Keine relevanten Gesetzesstellen gefunden."}) + "\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            user_prompt, citations = _build_ask_context(search_results, q, top_k)

            yield "event: search\ndata: " + json.dumps({"citations": citations, "count": len(citations)}) + "\n\n"

            provider = os.getenv("LLM_PROVIDER", "deepseek")
            model_used = provider

            if provider == "anthropic":
                from anthropic import Anthropic
                client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
                with client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    temperature=0.2,
                    system=LEX_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        yield "event: token\ndata: " + json.dumps(text) + "\n\n"
                model_used = "claude-sonnet-4-20250514"
            else:
                from openai import OpenAI
                client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                    base_url="https://api.deepseek.com",
                )
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    temperature=0.2,
                    stream=True,
                    messages=[
                        {"role": "system", "content": LEX_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield "event: token\ndata: " + json.dumps(delta) + "\n\n"
                model_used = "deepseek-chat"

            yield "event: done\ndata: " + json.dumps({"model": model_used}) + "\n\n"

        except Exception as e:
            yield "event: error\ndata: " + json.dumps({"message": str(e)}) + "\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@web_app.get("/api/legal/related/{doc_id:path}")
async def legal_related(doc_id: str):
    """Get related paragraphs from knowledge graph."""
    rag = LegalRAG()
    results = rag.get_related.remote(doc_id)
    return {"doc_id": doc_id, "related": results, "count": len(results)}


@web_app.get("/api/legal/stats")
async def legal_stats():
    """Index statistics."""
    rag = LegalRAG()
    return rag.stats.remote()


# ---------------------------------------------------------------------------
# Chat UI (served at /)
# ---------------------------------------------------------------------------
def _load_chat_html() -> str:
    """Load demo UI HTML, trying multiple paths (Modal vs local)."""
    candidates = [
        Path(__file__).parent / "src" / "static" / "demo_ui.html",
        Path("/src/static/demo_ui.html"),
        Path(__file__).parent / "static" / "demo_ui.html",
    ]
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8")
    return "<html><body><h1>UI file not found</h1></body></html>"

CHAT_HTML = _load_chat_html()


@web_app.get("/", response_class=HTMLResponse)
async def index():
    return CHAT_HTML


# ---------------------------------------------------------------------------
# Modal cron: weekly re-scrape + re-index (Saturday 02:00 UTC)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={VOLUME_PATH: VOLUME},
    secrets=[DEEPSEEK_SECRET, ANTHROPIC_SECRET],
    schedule=modal.Cron("0 2 * * 6"),
    timeout=3600,
)
def weekly_ingest():
    """Re-scrape all laws and update the index."""
    import asyncio

    sys.path.insert(0, "/")
    os.environ["LEGAL_RAG_STORAGE"] = str(VOLUME_PATH)

    from src.ingestion.rag_pipeline import LegalRAGPipeline
    from src.scrapers.gesetze_scraper import GesetzeScraper
    from src.processors.cleaner import clean
    from src.processors.metadata_extractor import extract as extract_metadata

    async def run():
        print("Starting weekly re-index...")
        all_docs = []
        async with GesetzeScraper() as scraper:
            docs = await scraper.scrape()
            all_docs.extend(docs)
        print(f"Scraped {len(all_docs)} documents")

        for doc in all_docs:
            if doc.get("inhalt"):
                doc["inhalt"] = clean(doc["inhalt"])
            doc = extract_metadata(doc)

        pipeline = LegalRAGPipeline()
        stats = await pipeline.insert_documents(all_docs)
        print(f"Indexed: {stats}")
        VOLUME.commit()
        return stats

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# Modal ASGI entry point
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={VOLUME_PATH: VOLUME},
    secrets=[DEEPSEEK_SECRET, ANTHROPIC_SECRET],
)
@modal.asgi_app()
def fastapi_app():
    return web_app
