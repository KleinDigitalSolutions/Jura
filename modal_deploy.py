"""Modal deployment for Legal RAG — hybrid search + LLM answer generation.

Deploy:  modal deploy modal_deploy.py
Test:    curl https://aliundmaggy--legal-rag-fastapi-app.modal.run/api/legal/search?q=Treu+und+Glauben
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

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

3. STRUKTURIERE deine Antwort zwingend nach dem IRAC-Schema (Markdown-Überschriften):
   ### 1. Issue (Rechtliche Fragestellung)
   - Präzise Zusammenfassung der juristischen Kernfrage.
   ### 2. Rule (Relevante Normen)
   - Nennung der primären und ergänzenden Rechtsgrundlagen (mit [Nummer]).
   ### 3. Analysis (Juristische Prüfung & Subsumtion)
   - Anwendung der Normen auf den Sachverhalt (beziehe dich auf den Mandantenfall).
   ### 4. Conclusion (Handlungsempfehlung)
   - Was muss der Mandant jetzt konkret tun? Welche Unterlagen fehlen ggf.?

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

        # Lazy-init query rewriter + classifier
        self.query_rewriter = None
        self.query_classifier = None
        print(f"LegalRAG loaded: {self.total_docs} docs in {time.monotonic() - t0:.1f}s")

    def _ensure_query_tools(self):
        """Initialize query rewriting / classification lazily."""
        if self.query_rewriter is None:
            from src.retrieval.query_rewriter import LegalQueryRewriter
            self.query_rewriter = LegalQueryRewriter()
        if self.query_classifier is None:
            from src.retrieval.query_classifier import LegalQueryClassifier
            self.query_classifier = LegalQueryClassifier()

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
    def enhanced_search(self, query: str, top_k: int = 10) -> dict:
        """Enhanced search with query rewriting + multi-query RRF."""
        self._ensure_query_tools()
        from src.retrieval.enhanced_search import EnhancedLegalSearch

        enhanced = EnhancedLegalSearch(
            searcher=self.searcher,
            rewriter=self.query_rewriter,
            classifier=self.query_classifier,
        )
        return enhanced.enhanced_search(query, top_k=top_k)

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

        # Failover mechanism: Try preferred provider first, then alternate
        provider = os.getenv("LLM_PROVIDER", "deepseek")
        providers = [provider]
        if provider == "deepseek":
            providers.append("anthropic")
        else:
            providers.append("deepseek")

        last_error = ""
        for p in providers:
            if p == "anthropic":
                try:
                    from anthropic import Anthropic
                    api_key = os.getenv("ANTHROPIC_API_KEY", "")
                    if not api_key:
                        continue
                    client = Anthropic(api_key=api_key)
                    response = client.messages.create(
                        model="claude-3-5-sonnet-latest",
                        max_tokens=2000,
                        temperature=0.2,
                        system=LEX_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    raw_answer = response.content[0].text.strip()
                    answer, warnings = _verify_citations(raw_answer, citations)
                    return {
                        "answer": answer,
                        "citations": citations,
                        "model": "claude-3-5-sonnet",
                        "citation_warnings": warnings,
                    }
                except Exception as e:
                    last_error = f"Anthropic error: {e}"
                    continue
            else:
                try:
                    from openai import OpenAI
                    api_key = os.getenv("DEEPSEEK_API_KEY", "")
                    if not api_key:
                        continue
                    client = OpenAI(
                        api_key=api_key,
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
                    raw_answer = response.choices[0].message.content.strip()
                    answer, warnings = _verify_citations(raw_answer, citations)
                    return {
                        "answer": answer,
                        "citations": citations,
                        "model": "deepseek-chat",
                        "citation_warnings": warnings,
                    }
                except Exception as e:
                    last_error = f"DeepSeek error: {e}"
                    continue

        return {
            "answer": f"KI-Schnittstellen (DeepSeek & Claude) sind derzeit nicht erreichbar. Bitte versuchen Sie es in Kürze erneut. ({last_error})",
            "citations": citations,
            "model": "error",
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
        stand = r.get("stand", "")

        # Inform the LLM about the context type
        ctx_info = ""
        ctype = r.get("context_type")
        if ctype == "neighbor":
            ctx_info = " (Nachbar-Paragraph / Kontext)"
        elif ctype == "citation":
            ctx_info = " (Zitierte/Verwandte Norm)"
        elif ctype == "citation_kg":
            ctx_info = " (Wissensgraph-Referenz)"

        # Warn if law text is potentially stale (>2 years old)
        stand_info = ""
        if stand:
            stand_info = f" [Stand: {stand}]"
            try:
                from datetime import datetime
                stand_date = datetime.strptime(stand, "%Y-%m-%d")
                days_old = (datetime.now() - stand_date).days
                if days_old > 730:
                    stand_info += " ⚠️ ÄLTER ALS 2 JAHRE — ÄNDERUNGEN MÖGLICH"
            except (ValueError, TypeError):
                pass

        if len(inhalt) > 3000:
            inhalt = inhalt[:3000] + "…"
            
        context_parts.append(f"{cite_id} {para} {abk}{ctx_info}{stand_info} — {titel}\n{inhalt}")
        citations.append({
            "id": cite_id,
            "gesetz": abk,
            "paragraph": para,
            "titel": titel,
            "url": r.get("url", ""),
            "score": r.get("rerank_score", r.get("score", 0)),
            "text_preview": inhalt[:500],
            "context_type": ctype or "primary",
            "stand": stand,
        })
    context = "\n\n---\n\n".join(context_parts)
    user_prompt = f"""FRAGE: {query}

GESETZESTEXTE (Primärtreffer & struktureller Kontext):
{context}

Analysiere die Frage präzise auf Basis der bereitgestellten Gesetzestexte. 
Beachte dabei besonders die Zusammenhänge zwischen den Paragraphen.
Zitiere jede Quelle mit ihrer [Nummer].
WICHTIG: Falls ein Gesetzestext mit "⚠️ ÄLTER ALS 2 JAHRE" markiert ist, 
weise den Mandanten darauf hin, dass Änderungen möglich sind und der 
aktuelle Stand geprüft werden muss."""
    return user_prompt, citations


def _verify_citations(answer: str, citations: list[dict]) -> tuple[str, list[str]]:
    """Post-process LLM answer to verify citations against provided context.

    Checks:
    1. [N] references in answer must correspond to provided citations
    2. § references in answer that are NOT in the provided context are flagged
    3. Appends a warning if ungrounded citations are detected

    Returns (possibly_annotated_answer, list_of_warning_strings).
    """
    import re as _re

    warnings: list[str] = []

    # 1. Check [N] bracket references — must be within range of provided citations
    valid_ids = {c["id"] for c in citations}
    bracket_refs = _re.findall(r'\[(\d+)\]', answer)
    for ref in bracket_refs:
        ref_id = f"[{ref}]"
        if ref_id not in valid_ids:
            warnings.append(f"Zitation [{ref}] nicht im bereitgestellten Kontext — mögliche Halluzination")

    # 2. Check § references in answer text that don't appear in any citation
    para_pattern = _re.compile(r'§+\s*(\d+[a-z]?)\s*(?:Abs\.|Absatz)?\s*(\d+)?', _re.IGNORECASE)
    answer_paras = set()
    for m in para_pattern.finditer(answer):
        para_num = m.group(1)
        answer_paras.add(para_num)

    cited_paras = set()
    for c in citations:
        p = c.get("paragraph", "")
        for m in para_pattern.finditer(p):
            cited_paras.add(m.group(1))

    ungrounded = answer_paras - cited_paras
    if ungrounded:
        for p in sorted(ungrounded):
            warnings.append(f"§ {p} in Antwort erwähnt, aber nicht in den bereitgestellten Gesetzestexten")

    # 3. Append warning to answer if any issues found
    if warnings:
        warning_block = "\n\n---\n⚠️ **Hinweis zur Quellenprüfung:**\n"
        for w in warnings:
            warning_block += f"- {w}\n"
        warning_block += "\nBitte überprüfen Sie diese Angaben mit einem Anwalt."
        answer = answer + warning_block

    return answer, warnings


# ---------------------------------------------------------------------------
# FastAPI web app
# ---------------------------------------------------------------------------
web_app = FastAPI(title="Legal RAG API")

# Input validation constants
MAX_QUERY_LENGTH = 500
MAX_TOP_K = 50


def _validate_query(q: str, top_k: int = 10) -> tuple[Optional[str], Optional[dict]]:
    """Validate and sanitize query input. Returns (sanitized_query, error_response).

    Checks: non-empty, max length, no control characters, top_k bounds.
    """
    if not q or not q.strip():
        return None, {"error": "Query parameter 'q' required"}

    q = q.strip()

    if len(q) > MAX_QUERY_LENGTH:
        return None, {"error": f"Query too long (max {MAX_QUERY_LENGTH} characters)"}

    # Strip control characters (keep newlines/tabs for legal text queries)
    import re as _re
    q = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', q)

    if not q:
        return None, {"error": "Query is empty after sanitization"}

    # Clamp top_k
    top_k = max(1, min(top_k, MAX_TOP_K))

    return q, None


@web_app.get("/api/legal/search")
async def legal_search(q: str = "", top_k: int = 10, rechtsgebiet: str = "", gesetz: str = ""):
    """Hybrid search endpoint."""
    q, err = _validate_query(q, top_k)
    if err:
        return err
    top_k = max(1, min(top_k, MAX_TOP_K))
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
    q, err = _validate_query(q, top_k)
    if err:
        return err
    top_k = max(1, min(top_k, MAX_TOP_K))
    rag = LegalRAG()
    result = rag.generate_answer.remote(q, top_k=top_k)
    return result


@web_app.get("/api/legal/ask/stream")
async def legal_ask_stream(q: str = "", top_k: int = 5):
    """Enhanced search + streaming LLM answer via Server-Sent Events."""
    q, err = _validate_query(q, top_k)
    if err:
        return err
    top_k = max(1, min(top_k, MAX_TOP_K))
    rag = LegalRAG()

    async def event_stream():
        try:
            enhanced_result = rag.enhanced_search.remote(q, top_k=top_k)
            search_results = enhanced_result.get("results", [])

            if not search_results:
                yield "event: error\ndata: " + json.dumps({"message": "Keine relevanten Gesetzesstellen gefunden."}) + "\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            context_k = min(len(search_results), top_k + 3)
            user_prompt, citations = _build_ask_context(search_results, q, context_k)

            yield "event: search\ndata: " + json.dumps({
                "citations": citations,
                "count": len(citations),
                "rewritten_queries": enhanced_result.get("rewritten_queries", [q]),
                "rechtsgebiet": enhanced_result.get("rechtsgebiet"),
                "retrieval_method": enhanced_result.get("retrieval_method", "full_fallback"),
            }) + "\n\n"

            # Failover mechanism for streaming
            provider = os.getenv("LLM_PROVIDER", "deepseek")
            providers = [provider]
            if provider == "deepseek":
                providers.append("anthropic")
            else:
                providers.append("deepseek")

            last_error = ""
            success = False
            answer_parts: list[str] = []
            for p in providers:
                try:
                    if p == "anthropic":
                        from anthropic import Anthropic
                        api_key = os.getenv("ANTHROPIC_API_KEY", "")
                        if not api_key: continue
                        client = Anthropic(api_key=api_key)
                        with client.messages.stream(
                            model="claude-3-5-sonnet-latest",
                            max_tokens=2000,
                            temperature=0.2,
                            system=LEX_SYSTEM_PROMPT,
                            messages=[{"role": "user", "content": user_prompt}],
                        ) as stream:
                            for text in stream.text_stream:
                                answer_parts.append(text)
                                yield "event: token\ndata: " + json.dumps(text) + "\n\n"
                        model_used = "claude-3-5-sonnet"
                        success = True
                        break
                    else:
                        from openai import OpenAI
                        api_key = os.getenv("DEEPSEEK_API_KEY", "")
                        if not api_key: continue
                        client = OpenAI(
                            api_key=api_key,
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
                                answer_parts.append(delta)
                                yield "event: token\ndata: " + json.dumps(delta) + "\n\n"
                        model_used = "deepseek-chat"
                        success = True
                        break
                except Exception as e:
                    last_error = str(e)
                    continue

            if not success:
                yield "event: error\ndata: " + json.dumps({"message": f"KI-Modelle nicht erreichbar. ({last_error})"}) + "\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            raw_answer = "".join(answer_parts)
            verified_answer, citation_warnings = _verify_citations(raw_answer, citations)
            if verified_answer != raw_answer:
                extra = (
                    verified_answer[len(raw_answer):]
                    if verified_answer.startswith(raw_answer)
                    else "\n\n" + verified_answer
                )
                yield "event: token\ndata: " + json.dumps(extra) + "\n\n"

            yield "event: done\ndata: " + json.dumps({
                "model": model_used,
                "citation_warnings": citation_warnings,
                "rewritten_queries": enhanced_result.get("rewritten_queries", [q]),
                "rechtsgebiet": enhanced_result.get("rechtsgebiet"),
                "retrieval_method": enhanced_result.get("retrieval_method", "full_fallback"),
            }) + "\n\n"

        except Exception as e:
            yield "event: error\ndata: " + json.dumps({"message": str(e)}) + "\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@web_app.get("/api/legal/ask/enhanced")
async def legal_ask_enhanced(q: str = "", top_k: int = 5):
    """Enhanced: query rewriting + multi-query RRF + LLM answer.

    Returns rewritten_queries and detected rechtsgebiet alongside
    the standard answer + citations.
    """
    q, err = _validate_query(q, top_k)
    if err:
        return err
    top_k = max(1, min(top_k, MAX_TOP_K))

    rag = LegalRAG()
    enhanced_result = rag.enhanced_search.remote(q, top_k=top_k)

    search_results = enhanced_result.get("results", [])
    if not search_results:
        return {
            "answer": "Keine relevanten Gesetzesstellen gefunden.",
            "citations": [],
            "rewritten_queries": enhanced_result.get("rewritten_queries", [q]),
            "rechtsgebiet": enhanced_result.get("rechtsgebiet"),
            "retrieval_method": enhanced_result.get("retrieval_method", "full_fallback"),
        }

    context_k = min(len(search_results), top_k + 3)
    user_prompt, citations = _build_ask_context(search_results, q, context_k)

    # Provider failover (gleiches Pattern wie generate_answer)
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    providers = [provider, "anthropic" if provider == "deepseek" else "deepseek"]

    last_error = ""
    answer_text = ""
    model_used = ""
    for p in providers:
        try:
            if p == "anthropic":
                from anthropic import Anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
                if not api_key:
                    continue
                client = Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=2000,
                    temperature=0.2,
                    system=LEX_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                answer_text = resp.content[0].text.strip()
                model_used = "claude-3-5-sonnet"
                break
            else:
                from openai import OpenAI
                api_key = os.getenv("DEEPSEEK_API_KEY", "")
                if not api_key:
                    continue
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": LEX_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                answer_text = resp.choices[0].message.content.strip()
                model_used = "deepseek-chat"
                break
        except Exception as e:
            last_error = f"{p}: {e}"
            continue

    if not answer_text:
        answer_text = f"KI-Schnittstellen nicht erreichbar. ({last_error})"
        model_used = "error"

    # Verify citations in answer (same guardrail as generate_answer)
    answer_text, citation_warnings = _verify_citations(answer_text, citations)

    return {
        "answer": answer_text,
        "citations": citations,
        "rewritten_queries": enhanced_result.get("rewritten_queries", [q]),
        "rechtsgebiet": enhanced_result.get("rechtsgebiet"),
        "retrieval_method": enhanced_result.get("retrieval_method", "full_fallback"),
        "model": model_used,
        "citation_warnings": citation_warnings,
    }


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
# Weekly scrape runs locally via launchd (Modal IPs blocked by gesetze-im-internet.de).
# See: scripts/weekly_scrape.sh + scripts/com.legal-rag.weekly-scrape.plist


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
