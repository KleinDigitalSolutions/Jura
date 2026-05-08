#!/usr/bin/env python3
"""Legal RAG Ingestion Bot — CLI entry point.

Usage:
    python main.py --run-all          Run all scrapers + ingestion
    python main.py --run-gesetze      Only German federal laws
    python main.py --run-urteile      Only court rulings
    python main.py --run-eurlex       Only EU law
    python main.py --schedule         Start weekly scheduler
    python main.py --stats            Show database statistics
"""
import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / f"ingestion_{datetime.now().strftime('%Y%m%d')}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
)

from src.config import RAG_STORAGE_DIR
from src.ingestion.rag_pipeline import STORAGE_DIR
from src.scrapers.gesetze_scraper import GesetzeScraper
from src.scrapers.eurlex_scraper import EurLexScraper
from src.scrapers.urteile_scraper import UrteileScraper
from src.processors.cleaner import clean
from src.processors.metadata_extractor import extract as extract_metadata
from src.processors.chunker import chunk_document
from src.ingestion.rag_pipeline import LegalRAGPipeline


async def run_full_pipeline(sources: list[str] = None) -> dict[str, int]:
    """Run scraping → cleaning → chunking → ingestion pipeline.
    sources: list of ['gesetze', 'eurlex', 'urteile'] or None for all.
    """
    if sources is None:
        sources = ["gesetze", "eurlex", "urteile"]

    all_raw_docs: list[dict] = []

    # Phase 1: Scrape
    if "gesetze" in sources:
        logger.info("=" * 50)
        logger.info("PHASE 1a: Scraping German federal laws...")
        try:
            async with GesetzeScraper() as scraper:
                docs = await scraper.scrape()
                all_raw_docs.extend(docs)
                logger.info(f"Gesetze: {len(docs)} documents scraped")
        except Exception as e:
            logger.error(f"Gesetze scraper failed: {e}")

    if "eurlex" in sources:
        logger.info("=" * 50)
        logger.info("PHASE 1b: Scraping EU law (EUR-Lex)...")
        try:
            async with EurLexScraper() as scraper:
                docs = await scraper.scrape()
                all_raw_docs.extend(docs)
                logger.info(f"EUR-Lex: {len(docs)} documents scraped")
        except Exception as e:
            logger.error(f"EUR-Lex scraper failed: {e}")

    if "urteile" in sources:
        logger.info("=" * 50)
        logger.info("PHASE 1c: Scraping court rulings...")
        try:
            async with UrteileScraper() as scraper:
                docs = await scraper.scrape()
                all_raw_docs.extend(docs)
                logger.info(f"Urteile: {len(docs)} documents scraped")
        except Exception as e:
            logger.error(f"Urteile scraper failed: {e}")

    if not all_raw_docs:
        logger.warning("No documents scraped, aborting pipeline")
        return {"inserted": 0, "failed": 0, "skipped": 0}

    logger.info(f"Total raw documents: {len(all_raw_docs)}")

    # Phase 2: Process (clean + metadata + chunk)
    logger.info("=" * 50)
    logger.info("PHASE 2: Processing documents...")
    chunked_docs: list[dict] = []

    for doc in all_raw_docs:
        try:
            # Clean
            if doc.get("inhalt"):
                doc["inhalt"] = clean(doc["inhalt"])
            if doc.get("volltext"):
                doc["volltext"] = clean(doc["volltext"])
            if doc.get("leitsatz"):
                doc["leitsatz"] = clean(doc["leitsatz"])

            # Extract metadata
            doc = extract_metadata(doc)

            # Chunk only long-form docs (Urteile); Gesetze paragraphs stay whole
            if doc.get("typ") == "urteil":
                chunks = chunk_document(doc)
                chunked_docs.extend(chunks)
            else:
                chunked_docs.append(doc)
        except Exception as e:
            logger.error(f"Processing failed for {doc.get('abkürzung', '')} {doc.get('paragraph', '')}: {e}")

    logger.info(f"Chunked into {len(chunked_docs)} documents from {len(all_raw_docs)} raw")

    # Phase 3: Ingest
    logger.info("=" * 50)
    logger.info("PHASE 3: Ingesting into RAG-Anything...")
    pipeline = LegalRAGPipeline()
    stats = await pipeline.insert_documents(chunked_docs)

    logger.info("=" * 50)
    logger.info(f"PIPELINE COMPLETE: {stats}")
    return stats


def show_stats():
    """Display current database statistics."""
    storage = STORAGE_DIR
    if not storage.exists():
        print(f"Storage directory not found: {storage}")
        return

    print(f"Storage directory: {storage}")
    for item in sorted(storage.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name}: {size_mb:.2f} MB")
        else:
            print(f"  {item.name}/")

    # Count records in knowledge graph
    graph_file = storage / "legal_graph.graphml"
    if graph_file.exists():
        content = graph_file.read_text()
        node_count = content.count("<node ")
        edge_count = content.count("<edge ")
        print(f"\nGraph nodes: ~{node_count}")
        print(f"Graph edges: ~{edge_count}")

    # Count documents
    docs_file = storage / "documents.json"
    if docs_file.exists():
        import json
        docs = json.loads(docs_file.read_text())
        print(f"Documents: {len(docs)}")

    # Qdrant size
    qdrant_dir = storage / "qdrant"
    if qdrant_dir.exists():
        total_size = sum(f.stat().st_size for f in qdrant_dir.rglob("*") if f.is_file())
        print(f"Qdrant index: {total_size / (1024 * 1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Legal RAG Ingestion Bot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-all", action="store_true", help="Run all scrapers + ingestion")
    group.add_argument("--run-gesetze", action="store_true", help="Only German federal laws")
    group.add_argument("--run-urteile", action="store_true", help="Only court rulings")
    group.add_argument("--run-eurlex", action="store_true", help="Only EU law")
    group.add_argument("--schedule", action="store_true", help="Start weekly scheduler")
    group.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--search", type=str, metavar="QUERY", help="Search indexed documents")
    parser.add_argument("--search-related", type=str, metavar="DOC_ID", help="Find related paragraphs via knowledge graph")
    parser.add_argument("--top-k", type=int, default=10, help="Results to return (default: 10)")
    parser.add_argument("--rechtsgebiet", type=str, help="Filter by legal area")
    parser.add_argument("--gesetz", type=str, help="Filter by law abbreviation (e.g. BGB)")

    args = parser.parse_args()

    if args.search:
        pipeline = LegalRAGPipeline()
        if not pipeline.indexer.load():
            print("No index found. Run --run-gesetze first.")
            return
        results = pipeline.search(
            args.search,
            top_k=args.top_k,
            rechtsgebiet=args.rechtsgebiet,
            gesetz=args.gesetz,
        )
        print(f"\nQuery: '{args.search}'")
        print(f"Results: {len(results)}\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['score']:.4f}] {r['label']}")
            text = r.get("inhalt", "") or r.get("volltext", "") or r.get("leitsatz", "")
            print(f"   {text[:200]}")
            if r.get("rechtsgebiet"):
                print(f"   Rechtsgebiet: {r['rechtsgebiet']}")
            print(f"   ID: {r.get('pid', '')}")
            print()
    elif args.search_related:
        pipeline = LegalRAGPipeline()
        if not pipeline.indexer.load():
            print("No index found. Run --run-gesetze first.")
            return
        # Find the doc by ID
        doc = pipeline.indexer._para_index.get(args.search_related)
        if not doc:
            print(f"Document not found: {args.search_related}")
            print("Format: ABK||paragraph  (e.g. BGB||§ 242)")
            return
        related = pipeline.searcher.get_related(doc)
        print(f"\nRelated to: {pipeline.indexer._doc_label(doc)}")
        print(f"Found {len(related)} references:\n")
        for i, r in enumerate(related[:args.top_k], 1):
            print(f"{i}. {r['label']}")
            text = r.get("inhalt", "") or r.get("volltext", "") or r.get("leitsatz", "")
            print(f"   {text[:150]}")
            print()
    elif args.stats:
        show_stats()
    elif args.schedule:
        from src.scheduler.cron import IngestionScheduler

        async def scheduled_job():
            await run_full_pipeline()

        scheduler = IngestionScheduler()
        scheduler.set_job(scheduled_job)
        scheduler.start()
    elif args.run_all:
        asyncio.run(run_full_pipeline())
    elif args.run_gesetze:
        asyncio.run(run_full_pipeline(sources=["gesetze"]))
    elif args.run_urteile:
        asyncio.run(run_full_pipeline(sources=["urteile"]))
    elif args.run_eurlex:
        asyncio.run(run_full_pipeline(sources=["eurlex"]))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
