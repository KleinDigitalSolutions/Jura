"""Pipeline: ingest structured legal documents into Qdrant + Knowledge Graph.

Uses BAAI/bge-m3 for dense (1024-dim) + learned sparse embeddings.
Weighted fusion (dense 0.7 / sparse 0.3) with min-max normalization.
bge-reranker-v2-m3 cross-encoder for precision reranking.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import torch
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config import BATCH_SIZE, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
STORAGE_DIR = Path(os.getenv("LEGAL_RAG_STORAGE", str(PROJECT_ROOT / "legal_rag_storage")))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_PATH = str(STORAGE_DIR / "qdrant")
GRAPH_PATH = str(STORAGE_DIR / "legal_graph.graphml")
DOCS_PATH = str(STORAGE_DIR / "documents.json")
COLLECTION_NAME = "legal_docs"
SPARSE_VECTOR_NAME = "lexical"

# Embedding model — bge-m3: 1024-dim dense + learned sparse
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_BATCH = int(os.getenv("EMBEDDING_BATCH", "32"))

# Reranker model
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

# ---------------------------------------------------------------------------
# § Reference parser — regex, no LLM
# ---------------------------------------------------------------------------
PARA_REF_PATTERN = re.compile(
    r"§+\s*\d+[a-z]?(?:\s*(?:Abs\.|Absatz|Absätze)\s*\d+(?:\s*(?:Nr\.|Nummer)\s*\d+)?)?"
    r"(?:\s*(?:S\.|Satz)\s*\d+)?"
    r"(?:\s*(?:Halbs\.|Halbsatz)\s*\d+)?"
    r"(?:\s*(?:Var\.|Variante)\s*\d+)?"
    r"(?:\s*(?:i\.\s*V\.\s*m\.|iVm|i\.V\.m\.|in Verbindung mit)\s*§+\s*\d+[a-z]?)?"
    r"(?:\s*(?:[,;]|und|i\.\s*V\.\s*m\.)\s*§+\s*\d+[a-z]?)*"
    r"(?:\s*(?:[A-Z][a-zäöüß]+(?:\s*[A-Z][a-zäöüß]+)*))?",
    re.IGNORECASE,
)

SIMPLER_REF = re.compile(r"§+\s*(\d+[a-z]?)", re.IGNORECASE)


def extract_para_refs(text: str) -> list[tuple[str, str]]:
    """Extract § references from text. Returns [(matched_text, para_number), ...]."""
    refs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in PARA_REF_PATTERN.finditer(text):
        full = match.group(0).strip()
        inner = SIMPLER_REF.findall(full)
        for num in inner:
            key = f"§ {num}"
            if key not in seen:
                seen.add(key)
                refs.append((full, num))
    return refs


def build_legal_graph(documents: list[dict]) -> nx.DiGraph:
    """Build knowledge graph from § references in document texts."""
    g = nx.DiGraph()

    para_index: dict[str, dict] = {}
    for doc in documents:
        pid = _doc_para_id(doc)
        abk = (doc.get("abkürzung", "") or "").upper()
        para = doc.get("paragraph", "")
        para_index[pid] = doc
        g.add_node(
            pid,
            label=para,
            abk=abk,
            titel=doc.get("titel", ""),
            paragraph_titel=doc.get("paragraph_titel", ""),
            rechtsgebiet=doc.get("rechtsgebiet", ""),
        )

    for doc in documents:
        src_id = _doc_para_id(doc)
        src_abk = (doc.get("abkürzung", "") or "").upper()
        text = doc.get("inhalt", "") or doc.get("volltext", "") or ""
        refs = extract_para_refs(text)

        for full_ref, ref_num in refs:
            tgt_id = f"{src_abk}||§ {ref_num}"
            matches = [pid for pid in para_index if pid.endswith(f"||§ {ref_num}")]
            for tgt in matches if matches else ([tgt_id] if tgt_id in para_index else []):
                if tgt != src_id and not g.has_edge(src_id, tgt):
                    g.add_edge(src_id, tgt, relation="verweist_auf", ref_text=full_ref)

    logger.info(f"Knowledge graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


def _doc_para_id(doc: dict) -> str:
    abk = (doc.get("abkürzung", "") or doc.get("gericht", "") or "").upper()
    para = doc.get("paragraph", "") or doc.get("aktenzeichen", "") or ""
    return f"{abk}||{para}"


# ---------------------------------------------------------------------------
# Embedder: bge-m3 — dense (1024-dim) + learned sparse
# ---------------------------------------------------------------------------
class LegalEmbedder:
    """bge-m3 embedding model — 1024-dim dense + learned sparse vectors."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        from FlagEmbedding import BGEM3FlagModel

        logger.info(f"Loading embedding model: {model_name}")
        use_fp16 = torch.cuda.is_available()
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.dim = EMBEDDING_DIM
        logger.info(f"Embedding dim: {self.dim} (dense + sparse), fp16={use_fp16}")

    def embed(self, texts: list[str], batch_size: int = EMBEDDING_BATCH) -> tuple[list[list[float]], list[SparseVector]]:
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        # bge-m3 does not L2-normalize — do it for cosine similarity
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        dense = dense / norms

        sparse: list[SparseVector] = []
        for lex_weights in output["lexical_weights"]:
            indices = list(lex_weights.keys())
            values = list(lex_weights.values())
            sparse.append(SparseVector(indices=indices, values=values))

        return dense.tolist(), sparse

    def embed_query(self, text: str) -> tuple[list[float], SparseVector]:
        dense_list, sparse_list = self.embed([text])
        return dense_list[0], sparse_list[0]


# ---------------------------------------------------------------------------
# Indexer: Qdrant (dense + sparse) + Knowledge Graph
# ---------------------------------------------------------------------------
class LegalIndexer:
    """Combined index: Qdrant vectors (dense + sparse) + NetworkX knowledge graph."""

    def __init__(self, storage_dir: Path = STORAGE_DIR, embedder: Optional[LegalEmbedder] = None):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder or LegalEmbedder()
        self.qdrant = QdrantClient(path=QDRANT_PATH)
        self.documents: list[dict] = []
        self.graph: Optional[nx.DiGraph] = None
        self._para_index: dict[str, dict] = {}

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedder.dim,
                    distance=Distance.COSINE,
                ),
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(full_scan_threshold=10000)
                    ),
                },
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME} (dense + sparse)")

    def _doc_text(self, doc: dict) -> str:
        parts: list[str] = []
        if doc.get("abkürzung"):
            parts.append(doc["abkürzung"])
        if doc.get("paragraph"):
            parts.append(doc["paragraph"])
        if doc.get("paragraph_titel"):
            parts.append(doc["paragraph_titel"])
        if doc.get("inhalt"):
            parts.append(doc["inhalt"])
        elif doc.get("volltext"):
            parts.append(doc["volltext"])
        elif doc.get("leitsatz"):
            parts.append(doc["leitsatz"])
        return " | ".join(parts)

    def _doc_label(self, doc: dict) -> str:
        abk = doc.get("abkürzung", "") or doc.get("gericht", "") or ""
        para = doc.get("paragraph", "") or doc.get("aktenzeichen", "") or ""
        t = doc.get("paragraph_titel", "") or ""
        if t:
            return f"{para} {abk} — {t}"
        return f"{para} {abk}"

    def index(self, documents: list[dict], incremental: bool = False) -> dict[str, int]:
        """Index documents: embed → Qdrant (dense+sparse), parse → Graph.

        incremental=True: append new docs to existing index without re-embedding old ones.
        incremental=False (default): clear + full rebuild from scratch.
        """
        stats = {"indexed": 0, "skipped": 0}
        start = time.monotonic()
        self._ensure_collection()

        if incremental and Path(DOCS_PATH).exists():
            # Load existing state — NOT re-embedded
            try:
                with open(DOCS_PATH, encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception:
                self.documents = []
            if Path(GRAPH_PATH).exists():
                try:
                    self.graph = nx.read_graphml(GRAPH_PATH)
                except Exception:
                    self.graph = None
            for i, doc in enumerate(self.documents):
                self._para_index[_doc_para_id(doc)] = {"doc": doc, "index": i}
            logger.info(f"Loaded {len(self.documents)} existing docs (incremental mode)")
        else:
            self.documents = []
            self.graph = None
            self._para_index = {}

        existing_count = len(self.documents)
        existing_pids = {_doc_para_id(d) for d in self.documents}

        # Filter valid + skip duplicates (by para_id)
        new_docs: list[dict] = []
        for doc in documents:
            inhalt = doc.get("inhalt", "") or doc.get("volltext", "") or doc.get("leitsatz", "")
            if not inhalt.strip() or inhalt.strip() in ("-", "(weggefallen)", "(aufgehoben)"):
                stats["skipped"] += 1
                continue
            pid = _doc_para_id(doc)
            if incremental and pid and pid in existing_pids:
                # Replace in existing documents list
                for j, edoc in enumerate(self.documents):
                    if _doc_para_id(edoc) == pid:
                        self.documents[j] = doc
                        break
                # Skip full re-embed; mark for lightweight update
                # For now: just skip to avoid re-embed cost
                # TODO: re-embed replaced docs + update Qdrant point
                stats["indexed"] += 1
                continue
            new_docs.append(doc)
            if pid:
                existing_pids.add(pid)
            stats["indexed"] += 1

        if not new_docs:
            logger.info(f"No new documents to embed (all {stats['indexed']} already known)")
            return stats

        new_total = len(new_docs)
        total_before = existing_count
        logger.info(f"Embedding {new_total} new docs (existing: {total_before})...")

        # --- Only embed NEW documents ---
        texts_for_embed = [self._doc_text(d) for d in new_docs]
        all_dense: list[list[float]] = []
        all_sparse: list[SparseVector] = []

        for i in range(0, new_total, EMBEDDING_BATCH):
            batch_texts = texts_for_embed[i : i + EMBEDDING_BATCH]
            dense, sparse = self.embedder.embed(batch_texts)
            all_dense.extend(dense)
            all_sparse.extend(sparse)
            if i % (EMBEDDING_BATCH * 20) == 0:
                logger.info(f"  Embedding: {min(i + EMBEDDING_BATCH, new_total)}/{new_total}")

        # Append new docs and upsert to Qdrant with correct point IDs
        chunk_size = 500
        for i in range(0, new_total, chunk_size):
            end = min(i + chunk_size, new_total)
            points = []
            for j in range(end - i):
                point_id = total_before + i + j
                doc = new_docs[i + j]
                self.documents.append(doc)
                self._para_index[_doc_para_id(doc)] = {"doc": doc, "index": len(self.documents) - 1}
                points.append(PointStruct(
                    id=point_id,
                    vector={
                        "": all_dense[i + j],
                        SPARSE_VECTOR_NAME: all_sparse[i + j],
                    },
                    payload={
                        "abk": doc.get("abkürzung", ""),
                        "paragraph": doc.get("paragraph", ""),
                        "titel": doc.get("titel", ""),
                        "rechtsgebiet": doc.get("rechtsgebiet", ""),
                        "quelle": doc.get("quelle", ""),
                        "stand": doc.get("stand", ""),
                    },
                ))
            self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

        # --- Rebuild Knowledge Graph ---
        if self.documents:
            self.graph = build_legal_graph(self.documents)

        self._save()
        elapsed = time.monotonic() - start
        logger.info(
            f"Indexing complete: {new_total} new docs embedded in {elapsed:.1f}s "
            f"({new_total / elapsed:.1f} docs/s), total: {len(self.documents)}"
        )
        return stats

    def _save(self) -> None:
        if self.graph:
            nx.write_graphml(self.graph, GRAPH_PATH)
        with open(DOCS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """Load persisted index. Returns False if no data."""
        if not Path(DOCS_PATH).exists():
            return False
        with open(DOCS_PATH, encoding="utf-8") as f:
            self.documents = json.load(f)
        if Path(GRAPH_PATH).exists():
            self.graph = nx.read_graphml(GRAPH_PATH)
        for i, doc in enumerate(self.documents):
            self._para_index[_doc_para_id(doc)] = {"doc": doc, "index": i}
        logger.info(
            f"Loaded index: {len(self.documents)} docs, "
            f"{self.graph.number_of_nodes() if self.graph else 0} nodes"
        )
        return True

    @property
    def total_docs(self) -> int:
        return len(self.documents)


# ---------------------------------------------------------------------------
# Searcher: Weighted Fusion (dense 0.7 + sparse 0.3) + Cross-encoder Reranker
# ---------------------------------------------------------------------------
class LegalSearcher:
    """Hybrid search: weighted dense+sparse fusion + cross-encoder reranker."""

    _QUERY_PARA = re.compile(r"§+\s*(\d+[a-z]?)", re.IGNORECASE)
    _QUERY_LAW = re.compile(
        r"\b(BGB|StGB|HGB|ZPO|StPO|GG|VwVfG|AktG|GmbHG?|InsO|FamFG|"
        r"BDSG|UrhG|MarkenG|PatG|BauGB|VwGO|AO|KStG|EStG|UStG|UmwG|WpHG|BetrVG|"
        r"SGB|KSchG|BVerfGG|TKG|WEG|EGBGB|BGBEG|UmwG|UWG|TTDSG|TDDDG)\b",
        re.IGNORECASE,
    )

    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3

    def __init__(self, indexer: LegalIndexer):
        self.indexer = indexer
        self.qdrant = indexer.qdrant
        self.embedder = indexer.embedder
        self._reranker = None  # lazy-loaded

    def _get_reranker(self):
        if self._reranker is None:
            from FlagEmbedding import FlagReranker

            use_fp16 = torch.cuda.is_available()
            logger.info(f"Loading reranker: {RERANKER_MODEL_NAME}")
            self._reranker = FlagReranker(RERANKER_MODEL_NAME, use_fp16=use_fp16)
        return self._reranker

    def _weighted_fusion(self, dense_results, sparse_results) -> list[tuple[int, float]]:
        """Min-max normalize each result set, then weighted merge (0.7 dense / 0.3 sparse)."""
        dense_scores: dict[int, float] = {}
        sparse_scores: dict[int, float] = {}

        if dense_results.points:
            d_scores = [p.score for p in dense_results.points]
            d_min, d_max = min(d_scores), max(d_scores)
            d_range = d_max - d_min if d_max != d_min else 1.0
            for p in dense_results.points:
                dense_scores[p.id] = (p.score - d_min) / d_range

        if sparse_results.points:
            s_scores = [p.score for p in sparse_results.points]
            s_min, s_max = min(s_scores), max(s_scores)
            s_range = s_max - s_min if s_max != s_min else 1.0
            for p in sparse_results.points:
                sparse_scores[p.id] = (p.score - s_min) / s_range

        all_ids = set(dense_scores) | set(sparse_scores)
        merged: list[tuple[int, float]] = []
        for doc_id in all_ids:
            d = dense_scores.get(doc_id, 0.0)
            s = sparse_scores.get(doc_id, 0.0)
            final = self.DENSE_WEIGHT * d + self.SPARSE_WEIGHT * s
            merged.append((doc_id, round(final, 4)))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Rerank candidates with cross-encoder."""
        if len(candidates) <= 1:
            return candidates

        reranker = self._get_reranker()
        pairs: list[tuple[str, str]] = []
        for c in candidates:
            text = c.get("inhalt", "") or c.get("volltext", "") or ""
            pairs.append((query, text[:512]))

        scores = reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = round(float(score), 4)

        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return candidates[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 10,
        rechtsgebiet: Optional[str] = None,
        gesetz: Optional[str] = None,
    ) -> list[dict]:
        """Hybrid search: §-pinning + weighted fusion (dense+sparse) + reranker."""
        # --- Parse query for explicit § references and law names ---
        q_paras = self._QUERY_PARA.findall(query)
        q_laws = [m.upper() for m in self._QUERY_LAW.findall(query)]

        # Auto-apply gesetz filter from query (e.g. "GmbH" → GmbHG)
        auto_gesetz = False
        if q_laws and not gesetz:
            gesetz = q_laws[0]
            auto_gesetz = True
            all_abks = {d.get("abkürzung", "").upper() for d in self.indexer.documents}
            if gesetz not in all_abks and gesetz + "G" in all_abks:
                gesetz = gesetz + "G"

        # --- Find exact § matches (pinned to top) ---
        pinned: list[dict] = []
        pinned_ids: set[int] = set()
        if q_paras:
            for idx, doc in enumerate(self.indexer.documents):
                para = (doc.get("paragraph", "") or "").strip()
                abk = (doc.get("abkürzung", "") or "").upper()
                if not para:
                    continue
                if not any(para.endswith(f" {n}") or para == f"§§ {n}" for n in q_paras):
                    continue
                if q_laws and not any(
                    abk == law or abk.startswith(law + " ") or abk.rstrip("G") == law
                    for law in q_laws
                ):
                    continue
                d = dict(doc)
                d["score"] = 1.0
                d["label"] = self.indexer._doc_label(doc)
                d["pid"] = _doc_para_id(doc)
                pinned.append(d)
                pinned_ids.add(idx)

        # --- Build Qdrant filter ---
        qf = None
        conditions = []
        if rechtsgebiet:
            conditions.append(FieldCondition(key="rechtsgebiet", match=MatchValue(value=rechtsgebiet)))
        if gesetz:
            # Match exact case from stored documents (Qdrant keyword matching is case-sensitive)
            gesetz_upper = gesetz.upper()
            exact_abk = None
            for d in self.indexer.documents:
                candidate = (d.get("abkürzung", "") or "").strip()
                if candidate.upper() == gesetz_upper:
                    exact_abk = candidate
                    break
            conditions.append(FieldCondition(key="abk", match=MatchValue(value=exact_abk or gesetz)))
        if conditions:
            qf = Filter(must=conditions)

        # --- Weighted Fusion: dense (0.7) + sparse (0.3) ---
        dense_vec, sparse_vec = self.embedder.embed_query(query)

        dense_response = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vec,
            limit=top_k * 3,
            query_filter=qf,
        )
        sparse_response = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=sparse_vec,
            using=SPARSE_VECTOR_NAME,
            limit=top_k * 3,
            query_filter=qf,
        )

        merged = self._weighted_fusion(dense_response, sparse_response)

        # Fallback: if auto-filter returns too few results, retry without filter
        # (handles "GmbH Insolvenz" — GmbHH narrows to GmbHG but Insolvenz is in InsO)
        if auto_gesetz and qf and len(merged) < top_k:
            logger.info(f"Auto-filter '{gesetz}' gave {len(merged)} results, retrying without filter")
            dense_no_filter = self.qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_vec,
                limit=top_k * 3,
            )
            sparse_no_filter = self.qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=sparse_vec,
                using=SPARSE_VECTOR_NAME,
                limit=top_k * 3,
            )
            merged = self._weighted_fusion(dense_no_filter, sparse_no_filter)

        # Build candidate list from merged results, excluding pinned docs
        candidates: list[dict] = []
        docs = self.indexer.documents
        for doc_id, score in merged:
            if doc_id in pinned_ids:
                continue
            if doc_id < len(docs):
                doc = dict(docs[doc_id])
                doc["score"] = round(score, 4)
                doc["doc_index"] = doc_id
                doc["label"] = self.indexer._doc_label(doc)
                doc["pid"] = _doc_para_id(doc)
                candidates.append(doc)

        # --- Structural Context Expansion (Sustainable Legal Logic) ---
        # For the top-5 primary candidates, pull in their immediate context
        expanded_candidates: list[dict] = []
        seen_ids: set[int] = pinned_ids.copy()
        
        # We start with the primary candidates
        for c in candidates:
            if c["doc_index"] not in seen_ids:
                expanded_candidates.append(c)
                seen_ids.add(c["doc_index"])

        # Expand: Adjacency (Neighbors) + Knowledge Graph (Citations)
        # Limit expansion to top 10 primary hits for maximum context
        primary_hits = candidates[:10]
        for primary in primary_hits:
            # 1. Adjacency Expansion (§ X-1, § X+1)
            idx = primary["doc_index"]
            for offset in [-1, 1]:
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(docs) and neighbor_idx not in seen_ids:
                    neighbor = docs[neighbor_idx]
                    if neighbor.get("abkürzung") == primary.get("abkürzung"):
                        d = dict(neighbor)
                        d["score"] = primary["score"] * 0.95  # Minimal penalty
                        d["doc_index"] = neighbor_idx
                        d["label"] = self.indexer._doc_label(neighbor)
                        d["pid"] = _doc_para_id(neighbor)
                        d["context_type"] = "neighbor"
                        expanded_candidates.append(d)
                        seen_ids.add(neighbor_idx)

            # 2. Relational Expansion (Knowledge Graph)
            graph_neighbors = self.get_related(primary)
            for gn in graph_neighbors:
                gn_pid = gn.get("pid")
                # Fast lookup using _para_index
                indexed_hit = self.indexer._para_index.get(gn_pid)
                if indexed_hit:
                    idx = indexed_hit["index"]
                    if idx not in seen_ids:
                        d = dict(indexed_hit["doc"])
                        d["score"] = primary["score"] * 0.90  # Slight penalty
                        d["doc_index"] = idx
                        d["label"] = self.indexer._doc_label(d)
                        d["pid"] = gn_pid
                        d["context_type"] = "citation"
                        expanded_candidates.append(d)
                        seen_ids.add(idx)

        # --- Rerank Expanded Candidates ---
        # The reranker will now see § 15a AND § 15b and can decide which is better
        reranked = self._rerank(query, expanded_candidates, top_k)

        # Prepend pinned exact matches
        results = pinned + reranked
        return results[:top_k]

    def get_related(self, doc: dict) -> list[dict]:
        """Get related paragraphs from knowledge graph (both directions)."""
        if not self.indexer.graph:
            return []
        pid = _doc_para_id(doc)
        if pid not in self.indexer.graph:
            return []
        related: list[dict] = []
        seen: set[str] = {pid}
        for src, tgt in list(self.indexer.graph.out_edges(pid)) + list(self.indexer.graph.in_edges(pid)):
            neighbor = tgt if src == pid else src
            if neighbor in seen:
                continue
            seen.add(neighbor)
            if neighbor in self.indexer._para_index:
                r = dict(self.indexer._para_index[neighbor])
                r["label"] = self.indexer._doc_label(r)
                r["pid"] = neighbor
                related.append(r)
        return related


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------
class LegalRAGPipeline:
    """Orchestrates indexing and searching for legal documents.

    Usage:
        pipeline = LegalRAGPipeline()
        await pipeline.insert_documents(docs)     # Scrape → index
        results = pipeline.search("Treu und Glauben")  # Query
    """

    def __init__(self):
        self.embedder = LegalEmbedder()
        self.indexer = LegalIndexer(embedder=self.embedder)
        self.searcher = LegalSearcher(self.indexer)
        self._stats: dict[str, int] = {"inserted": 0, "failed": 0, "skipped": 0}

    async def insert_documents(self, documents: list[dict], incremental: bool = True) -> dict[str, int]:
        """Insert documents into all indexes (Qdrant + Graph).

        incremental=True (default): only new docs are embedded. Safe for partial scrapes.
        incremental=False: full rebuild (used by rebuild_clean.py after clearing Qdrant).
        """
        if not documents:
            logger.warning("No documents to insert")
            return self._stats

        mode = "incremental" if incremental else "full rebuild"
        total_in = len(documents)
        logger.info(f"Inserting {total_in} documents ({mode})...")
        start = time.monotonic()

        valid: list[dict] = []
        for doc in documents:
            text = doc.get("inhalt", "") or doc.get("volltext", "") or doc.get("leitsatz", "")
            if not text.strip():
                self._stats["skipped"] += 1
                continue
            valid.append(doc)

        if not valid:
            return self._stats

        stats = self.indexer.index(valid, incremental=incremental)
        self._stats["inserted"] = stats["indexed"]
        self._stats["skipped"] += stats["skipped"]

        elapsed = time.monotonic() - start
        logger.info(
            f"Insertion complete: {self._stats['inserted']} indexed, "
            f"{self._stats['skipped']} skipped in {elapsed:.1f}s"
        )
        return self._stats

    def search(
        self,
        query: str,
        top_k: int = 10,
        rechtsgebiet: Optional[str] = None,
        gesetz: Optional[str] = None,
    ) -> list[dict]:
        return self.searcher.search(
            query, top_k=top_k, rechtsgebiet=rechtsgebiet, gesetz=gesetz
        )

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def total_docs(self) -> int:
        return self.indexer.total_docs
