"""Legal-specific document chunking.

Key principles:
- Gesetze: NEVER split mid-paragraph. One Chunk = One §.
- Urteile: Leitsatz own chunk. Volltext 1000-token / 200 overlap.
- Metadata always prepended as text prefix (LightRAG has no native metadata field).
"""
from typing import Optional


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for German text."""
    return len(text) // 4


def chunk_gesetz(doc: dict, max_chunk_tokens: int = 2000) -> list[dict]:
    """Chunk a law paragraph document.
    1 paragraph = 1 chunk. Long paragraphs (>max_chunk_tokens) get sub-chunks.
    """
    inhalt = doc.get("inhalt", "")
    meta_prefix = _build_meta_prefix(doc)
    chunks: list[dict] = []

    token_count = estimate_tokens(inhalt)

    if token_count <= max_chunk_tokens:
        # Single chunk: paragraph fits
        chunks.append({**doc, "inhalt": f"{meta_prefix}\n\n{inhalt}"})
    else:
        # Split into sub-chunks by sentence/paragraph boundary
        sub_chunks = _split_text_by_boundary(inhalt, max_chunk_tokens)
        for i, sub in enumerate(sub_chunks):
            chunks.append({
                **doc,
                "inhalt": f"{meta_prefix} [Teil {i + 1}/{len(sub_chunks)}]\n\n{sub}",
            })

    return chunks


def chunk_urteil(doc: dict, max_chunk_tokens: int = 1000, overlap_tokens: int = 200) -> list[dict]:
    """Chunk a court ruling document.
    - Leitsatz always as own chunk
    - Volltext in 1000-token chunks with 200 overlap
    """
    meta_prefix = _build_meta_prefix(doc)
    chunks: list[dict] = []

    leitsatz = doc.get("leitsatz", "")
    volltext = doc.get("volltext", "")

    # Leitsatz as dedicated chunk
    if leitsatz.strip():
        chunks.append({
            **doc,
            "inhalt": f"{meta_prefix}\n[LEITSATZ]\n\n{leitsatz.strip()}",
        })

    # Split Volltext into sections: Tatbestand, Entscheidungsgründe
    tatsache, entscheidung = _split_urteil_sections(volltext)

    # Chunk Tatbestand
    if tatsache.strip():
        for i, chunk_text in enumerate(_sliding_chunks(tatsache, max_chunk_tokens, overlap_tokens)):
            chunks.append({
                **doc,
                "inhalt": f"{meta_prefix} [Tatbestand Teil {i + 1}]\n\n{chunk_text}",
            })

    # Chunk Entscheidungsgründe
    if entscheidung.strip():
        for i, chunk_text in enumerate(_sliding_chunks(entscheidung, max_chunk_tokens, overlap_tokens)):
            chunks.append({
                **doc,
                "inhalt": f"{meta_prefix} [Entscheidungsgründe Teil {i + 1}]\n\n{chunk_text}",
            })

    # Fallback: if no sections found, chunk the whole text
    if not chunks:
        for i, chunk_text in enumerate(_sliding_chunks(volltext, max_chunk_tokens, overlap_tokens)):
            chunks.append({
                **doc,
                "inhalt": f"{meta_prefix} [Teil {i + 1}]\n\n{chunk_text}",
            })

    return chunks


def chunk_document(doc: dict) -> list[dict]:
    """Route to appropriate chunker based on document type."""
    typ = doc.get("typ", "")

    if typ == "gesetz" or typ == "eu_verordnung":
        return chunk_gesetz(doc)
    elif typ == "urteil":
        return chunk_urteil(doc)
    else:
        # Generic: treat as single chunk
        meta_prefix = _build_meta_prefix(doc)
        return [{**doc, "inhalt": f"{meta_prefix}\n\n{doc.get('inhalt', '')}"}]


def _build_meta_prefix(doc: dict) -> str:
    """Build metadata text prefix to prepend to chunk content.
    LightRAG has no native metadata field, so we embed it in the text.
    """
    parts: list[str] = []

    if doc.get("typ"):
        parts.append(f"Typ: {doc['typ']}")
    if doc.get("abkürzung"):
        parts.append(f"Gesetz: {doc['abkürzung']}")
    if doc.get("titel"):
        parts.append(f"Titel: {doc['titel']}")
    if doc.get("paragraph"):
        parts.append(f"Paragraph: {doc['paragraph']}")
    if doc.get("paragraph_titel"):
        parts.append(f"Abschnitt: {doc['paragraph_titel']}")
    if doc.get("gericht"):
        parts.append(f"Gericht: {doc['gericht']}")
    if doc.get("aktenzeichen"):
        parts.append(f"Aktenzeichen: {doc['aktenzeichen']}")
    if doc.get("datum"):
        parts.append(f"Datum: {doc['datum']}")
    if doc.get("rechtsgebiet"):
        parts.append(f"Rechtsgebiet: {doc['rechtsgebiet']}")
    if doc.get("stand"):
        parts.append(f"Stand: {doc['stand']}")
    if doc.get("quelle"):
        parts.append(f"Quelle: {doc['quelle']}")
    if doc.get("url"):
        parts.append(f"URL: {doc['url']}")

    return " | ".join(parts)


def _split_text_by_boundary(text: str, max_tokens: int) -> list[str]:
    """Split text at legal structure boundaries, respecting max_tokens per chunk.

    German legal text structure differs from prose:
    - Numbered lists: "1. ...; 2. ...; 3. ..." should stay together
    - Absatz markers: "(1) ... (2) ..." are natural split points
    - Semicolons often separate legal provisions within a sentence
    - Simple period-based splitting breaks mid-provision

    Priority split points (strongest first):
    1. Absatz markers: "(1)", "(2)", etc.
    2. Numbered list items: "1.", "2.", etc. (but only legal-style, not sentence-start)
    3. Double newlines (paragraph breaks)
    4. Semicolons followed by a capital letter (provision boundary)
    5. Sentence boundaries as last resort
    """
    import re

    # Strategy: split into "legal segments" first, then merge into chunks
    # A legal segment is the text between two high-priority split points.

    # Pattern for Absatz markers: (1), (2), (1a), etc.
    absatz_pattern = re.compile(r'(?=\(\d+[a-z]?\)\s)', re.IGNORECASE)

    # Pattern for numbered list items: "1. ", "2. " at start of line or after semicolon
    # But NOT "1. Einleitung..." which is a normal sentence start
    list_pattern = re.compile(r'(?<=;)\s*(?=\d+\.\s+[A-ZÄÖÜ])|(?<=\n)\s*(?=\d+\.\s+[A-ZÄÖÜ])')

    # Try Absatz splitting first (most reliable for German law)
    segments = absatz_pattern.split(text)

    # If we only got 1 segment, try numbered list splitting
    if len(segments) <= 1:
        segments = list_pattern.split(text)

    # If still 1 segment, try double-newline splitting
    if len(segments) <= 1:
        segments = re.split(r'\n\s*\n', text)

    # If still 1 segment, fall back to semicolon boundaries
    if len(segments) <= 1:
        segments = re.split(r';\s*(?=[A-ZÄÖÜ])', text)

    # If still 1 segment, fall back to sentence splitting (old behavior)
    if len(segments) <= 1:
        segments = re.split(r'(?<=[.!?])\s+', text)

    # Merge segments into chunks respecting max_tokens
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for segment in segments:
        seg_tokens = estimate_tokens(segment)
        if current_tokens + seg_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = [segment]
            current_tokens = seg_tokens
        else:
            current.append(segment)
            current_tokens += seg_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _sliding_chunks(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Create sliding window chunks with token overlap."""
    words = text.split()
    chunks: list[str] = []
    i = 0

    chunk_words = chunk_tokens  # approximate, ~1 word = 1 token for German
    overlap_words = overlap_tokens

    while i < len(words):
        chunk = " ".join(words[i:i + chunk_words])
        if chunk:
            chunks.append(chunk)
        i += chunk_words - overlap_words
        if i >= len(words):
            break

    return chunks


def _split_urteil_sections(volltext: str) -> tuple[str, str]:
    """Split court ruling text into Tatbestand and Entscheidungsgründe sections."""
    import re

    tatsache = ""
    entscheidung = ""

    # Common patterns for ruling section markers
    tatsache_pattern = re.compile(
        r'(?:^|\n)\s*(?:Tatbestand|T a t b e s t a n d|Sachverhalt)\s*(?:$|\n)',
        re.IGNORECASE
    )
    entscheidung_pattern = re.compile(
        r'(?:^|\n)\s*(?:Entscheidungsgründe|E n t s c h e i d u n g s g r ü n d e|Gründe|Begründung)\s*(?:$|\n)',
        re.IGNORECASE
    )

    t_match = tatsache_pattern.search(volltext)
    e_match = entscheidung_pattern.search(volltext)

    if t_match and e_match:
        tatsache = volltext[t_match.start():e_match.start()]
        entscheidung = volltext[e_match.start():]
    elif e_match:
        tatsache = volltext[:e_match.start()]
        entscheidung = volltext[e_match.start():]
    elif t_match:
        tatsache = volltext[t_match.start():]
        entscheidung = volltext[:t_match.start()]
    else:
        # No clear sections found
        tatsache = volltext
        entscheidung = ""

    return tatsache.strip(), entscheidung.strip()
