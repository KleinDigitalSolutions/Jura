"""Enhanced legal search orchestrator.

Combines query classification → query rewriting → multi-query RRF search
into a single pipeline with graceful degradation at every stage.
"""
import logging
import re
from typing import Optional

from src.ingestion.rag_pipeline import LegalSearcher
from src.retrieval.legal_quality import apply_legal_quality, plan_to_dict
from src.retrieval.query_classifier import LegalQueryClassifier
from src.retrieval.query_rewriter import LegalQueryRewriter

logger = logging.getLogger(__name__)

QUERY_LAW_PATTERN = re.compile(
    r"\b(BGB|StGB|HGB|ZPO|StPO|GG|VwVfG|AktG|GmbHG?|InsO|FamFG|"
    r"BDSG|UrhG|MarkenG|PatG|BauGB|VwGO|AO|KStG|EStG|UStG|UmwG|WpHG|BetrVG|"
    r"SGB|KSchG|BVerfGG|TKG|WEG|EGBGB|BGBEG|UWG|TTDSG|TDDDG)\b",
    re.IGNORECASE,
)


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
                - retrieval_plan: deterministic legal quality plan
                - source_audit: accepted/rejected/injected source audit
        """
        result = {
            "results": [],
            "rewritten_queries": [query],
            "rechtsgebiet": None,
            "retrieval_method": "full_fallback",
            "original_query": query,
            "retrieval_plan": None,
            "source_audit": None,
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
        query_laws = {m.upper() for m in QUERY_LAW_PATTERN.findall(query)}

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
        # Add a small amount of cited context after the primary results. The
        # answer builder can include these extra items as supporting citations.
        if result["results"]:
            try:
                kg_expanded = list(result["results"])
                original_count = len(kg_expanded)
                seen_pids: set[str] = {
                    r.get("pid", "") for r in kg_expanded if r.get("pid")
                }
                for doc in result["results"][:3]:
                    try:
                        related = self.searcher.get_related(doc)
                    except Exception:
                        continue
                    primary_score = doc.get("score", 0)
                    primary_abk = doc.get("abkürzung", "")
                    primary_rechtsgebiet = doc.get("rechtsgebiet", "")
                    for r_doc in related:
                        pid = r_doc.get("pid", "")
                        text = (
                            r_doc.get("inhalt", "")
                            or r_doc.get("volltext", "")
                            or r_doc.get("leitsatz", "")
                        )
                        related_abk = r_doc.get("abkürzung", "")
                        related_rechtsgebiet = r_doc.get("rechtsgebiet", "")
                        if query_laws and related_abk.upper() not in query_laws:
                            continue
                        same_law = primary_abk and related_abk and primary_abk == related_abk
                        same_area = (
                            primary_rechtsgebiet
                            and related_rechtsgebiet
                            and primary_rechtsgebiet == related_rechtsgebiet
                        )
                        if not (same_law or same_area):
                            continue
                        if pid and text and pid not in seen_pids:
                            new_doc = dict(r_doc)
                            new_doc["score"] = round(primary_score * 0.90, 4)
                            new_doc["context_type"] = "citation_kg"
                            kg_expanded.append(new_doc)
                            seen_pids.add(pid)
                result["results"] = kg_expanded[: top_k + 3]
                result["kg_expanded_count"] = max(0, len(kg_expanded) - original_count)
            except Exception:
                logger.exception("Knowledge Graph expansion failed")

        # Step 5: deterministic legal quality layer
        # Profiles add mandatory norms for known issue classes and remove
        # known false-positive sources before answer generation.
        if result["results"]:
            try:
                qualified, plan, audit = apply_legal_quality(
                    query=query,
                    results=result["results"],
                    searcher=self.searcher,
                    rechtsgebiet=rechtsgebiet,
                    top_k=top_k,
                )
                result["results"] = qualified
                result["retrieval_plan"] = plan_to_dict(plan)
                result["source_audit"] = audit.as_dict()
            except Exception:
                logger.exception("Legal quality layer failed")

        return result
