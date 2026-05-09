#!/bin/bash
# Weekly legal RAG scrape + index + upload to Modal Volume.
# Runs Saturdays 04:00 CET via launchd.
# Logs: logs/weekly_scrape_YYYYMMDD.log
set -euo pipefail

PROJECT_DIR="/Users/bucci369/legal-rag-ingestion"
LOG_FILE="$PROJECT_DIR/logs/weekly_scrape_$(date +%Y%m%d).log"
mkdir -p "$PROJECT_DIR/logs"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Weekly scrape starting: $(date) ==="

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "Starting Python pipeline..."
python -c "
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, '.')
from src.ingestion.rag_pipeline import LegalRAGPipeline, STORAGE_DIR
from src.scrapers.gesetze_scraper import GesetzeScraper
from src.scrapers.urteile_scraper import UrteileScraper
from src.processors.cleaner import clean
from src.processors.metadata_extractor import extract as extract_metadata
from src.processors.chunker import chunk_document

async def run():
    all_docs = []

    # Gesetze
    async with GesetzeScraper() as s:
        docs = await s.scrape()
        all_docs.extend(docs)
        print(f'Gesetze: {len(docs)} documents scraped')

    # Urteile — all 7 courts
    async with UrteileScraper(
        courts=['BGH','BVerfG','BVerwG','BFH','BAG','BSG','BPatG'],
        max_per_court=0,
    ) as s:
        docs = await s.scrape()
        for d in docs:
            d = chunk_document(d)
        all_docs.extend(docs)
        print(f'Urteile: {len(docs)} documents scraped (after chunking)')

    # Clean + metadata
    for doc in all_docs:
        if doc.get('inhalt'):
            doc['inhalt'] = clean(doc['inhalt'])
        doc = extract_metadata(doc)

    print(f'Total raw docs: {len(all_docs)}')

    # Incremental index (embeds only new docs)
    pipeline = LegalRAGPipeline()
    stats = await pipeline.insert_documents(all_docs, incremental=True)
    print(f'Index result: {json.dumps(stats)}')

asyncio.run(run())
"

echo ""
echo "Pipeline done. Uploading to Modal Volume..."

# Upload changed files to Modal Volume
modal volume put legal-rag-data documents.json legal_rag_storage/documents.json
modal volume put legal-rag-data legal_graph.graphml legal_rag_storage/legal_graph.graphml
modal volume put legal-rag-data qdrant/meta.json legal_rag_storage/qdrant/meta.json
modal volume put legal-rag-data qdrant/collection/legal_docs/storage.sqlite legal_rag_storage/qdrant/collection/legal_docs/storage.sqlite

echo ""
echo "=== Weekly scrape complete: $(date) ==="
