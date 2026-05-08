"""Rebuild index without (weggefallen) placeholder paragraphs."""
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("LEGAL_RAG_STORAGE", str(Path(__file__).parent / "legal_rag_storage"))

from src.ingestion.rag_pipeline import LegalRAGPipeline, STORAGE_DIR

# Load existing documents
with open(STORAGE_DIR / "documents.json") as f:
    docs = json.load(f)

print(f"Before: {len(docs)} docs")

# Count and remove weggefallen
weg_count = sum(
    1 for d in docs
    if (d.get("inhalt", "") or "").strip() in ("-", "(weggefallen)", "(aufgehoben)")
)
print(f"Weggefallen/aufgehoben: {weg_count}")

def _is_junk(doc: dict) -> bool:
    inhalt = (doc.get("inhalt", "") or "").strip()
    para = (doc.get("paragraph", "") or "").strip()
    abk = (doc.get("abkürzung", "") or "").strip()

    # (weggefallen) / (aufgehoben) placeholder paragraphs
    # Also catch "-" followed by repeal notes (e.g. "- \n§ 219c: Aufgeh. durch...")
    if inhalt in ("-", "(weggefallen)", "(aufgehoben)") or inhalt.startswith("-"):
        return True

    # BJNR*/BJNG* = XML node IDs (preamble, not real paragraphs)
    if para.startswith("BJNR") or para.startswith("BJNG"):
        return True

    # Table of contents entries
    if "Inhaltsübersicht" in para or "Inhaltsverzeichnis" in para:
        return True

    # ./ prefix = unofficial table-of-contents (nichtamtliches Inhaltsverzeichnis)
    if abk.startswith("./"):
        return True

    return False

clean_docs = [d for d in docs if not _is_junk(d)]
bjnr_count = sum(1 for d in docs if ((d.get("paragraph","") or "").startswith("BJNR") or (d.get("paragraph","") or "").startswith("BJNG")))
toc_count = sum(1 for d in docs if "Inhaltsübersicht" in (d.get("paragraph","") or "") or "Inhaltsverzeichnis" in (d.get("paragraph","") or ""))
dot_slash = sum(1 for d in docs if (d.get("abkürzung","") or "").startswith("./"))
print(f"Filtered: {weg_count} weggefallen, {bjnr_count} BJNR*, {toc_count} TOC, {dot_slash} ./ prefixed")
print(f"After: {len(clean_docs)} docs")

# Save filtered docs
with open(STORAGE_DIR / "documents.json", "w") as f:
    json.dump(clean_docs, f, ensure_ascii=False, indent=2)

# Clear old index files
for fname in ["legal_graph.graphml"]:
    p = STORAGE_DIR / fname
    if p.exists():
        p.unlink()

# Clear Qdrant
qd = STORAGE_DIR / "qdrant"
if qd.exists():
    shutil.rmtree(qd)

print("Old index cleared. Rebuilding...")

# Rebuild from cleaned docs
pipeline = LegalRAGPipeline()
stats = asyncio.run(pipeline.insert_documents(clean_docs, incremental=False))
print(f"Rebuild done: {stats}")
