"""Enhanced legal search orchestrator.

Combines query classification → query rewriting → multi-query RRF search
into a single pipeline with graceful degradation at every stage.
"""
import logging
from typing import Optional

from src.ingestion.rag_pipeline import LegalSearcher
from src.retrieval.query_classifier import LegalQueryClassifier
from src.retrieval.query_rewriter import LegalQueryRewriter

logger = logging.getLogger(__name__)


class EnhancedLegalSearch:
    """Orchestrates classification, rewriting, and multi-query search.

    The pipeline:
        1. QueryClassifier — fast keyword-based Rechtsgebiet detection
        2. LegalQueryRewriter — LLM transforms query into 3 legal variants
        3. LegalSearcher.search_multi_query — RRF fusion across variants

    Every step degrades gracefully:
        - Classifier fails → rechtsgebiet = None, search proceeds
        - Rewriter fails → falls back to single query search
        - Multi-query search fails → falls back to single search
    """

    def __init__(
        self,
        searcher: LegalSearcher,
        rewriter: Optional[LegalQueryRewriter] = None,
        classifier: Optional[LegalQueryClassifier] = None,
    ):
        self.searcher = searcher
        self.rewriter = rewriter or LegalQueryRewriter()
        self.classifier = classifier or LegalQueryClassifier()

    def enhanced_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> dict:
        """Run the full enhanced search pipeline.

        Args:
            query: User query in natural language.
            top_k: Number of results to return.

        Returns:
            dict with keys:
                - results: list of result dicts (same format as search())
                - rewritten_queries: [original, ...variant1, variant2, variant3]
                - rechtsgebiet: detected legal area or None
                - retrieval_method: "enhanced" | "fallback" | "full_fallback"
                - original_query: the input query
        """
        result = {
            "results": [],
            "rewritten_queries": [query],
            "rechtsgebiet": None,
            "retrieval_method": "full_fallback",
            "original_query": query,
        }

        # Step 1: Classify Rechtsgebiet
        rechtsgebiet = None
        try:
            rechtsgebiet = self.classifier.classify(query)
        except Exception:
            logger.exception("Query classification failed")
        result["rechtsgebiet"] = rechtsgebiet

        # Step 2: Rewrite query
        all_queries = [query]
        try:
            variants = self.rewriter.rewrite(query, rechtsgebiet=rechtsgebiet)
            # Deduplicate: include original + non-duplicate variants
            seen = {query.lower()}
            for v in variants:
                if v.lower() not in seen:
                    seen.add(v.lower())
                    all_queries.append(v)
        except Exception:
            logger.exception("Query rewriting failed, falling back to single query")

        result["rewritten_queries"] = all_queries

        # Step 3: Search — multi-query or single
        if len(all_queries) > 1:
            try:
                results = self.searcher.search_multi_query(
                    all_queries,
                    top_k=top_k,
                )
                result["results"] = results
                result["retrieval_method"] = "enhanced"
            except Exception:
                logger.exception(
                    "Multi-query search failed, falling back to single query"
                )
                # Fall through to single-query fallback

        # Fallback: single query search
        if not result["results"]:
            try:
                results = self.searcher.search(query, top_k=top_k)
                result["results"] = results
                result["retrieval_method"] = (
                    "fallback" if result["retrieval_method"] == "full_fallback"
                    else result["retrieval_method"]
                )
                # If multi-query failed but we have results, mark as fallback
                if result["retrieval_method"] == "enhanced":
                    result["retrieval_method"] = "fallback"
            except Exception:
                logger.exception("Fallback search also failed")

        # Step 4: Knowledge Graph expansion
        # For each top-3 result, find related paragraphs via the 965K-edge graph
        # and append them as citation context (score * 0.90, matching existing pattern)
        if result["results"]:
            try:
                kg_expanded = list(result["results"])
                seen_pids: set[str] = {
                    r.get("pid", "") for r in kg_expanded if r.get("pid")
                }
                for doc in result["results"][:3]:
                    try:
                        related = self.searcher.get_related(doc)
                    except Exception:
                        continue
                    primary_score = doc.get("score", 0)
                    for r_doc in related:
                        pid = r_doc.get("pid", "")
                        if pid and pid not in seen_pids:
                            new_doc = dict(r_doc)
                            new_doc["score"] = round(primary_score * 0.90, 4)
                            new_doc["context_type"] = "citation_kg"
                            kg_expanded.append(new_doc)
                            seen_pids.add(pid)
                result["results"] = kg_expanded
                result["kg_expanded_count"] = len(kg_expanded) - len(result["results"])
            except Exception:
                logger.exception("Knowledge Graph expansion failed")

        return result
