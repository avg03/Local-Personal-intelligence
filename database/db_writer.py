"""
db_writer.py
Writes ingestion pipeline output into the SQLite schema:
resources, chunks, interactions.

concepts + relationships are NOT written here — those need a Gemma
summarization/concept-extraction step that doesn't exist yet. Don't
call save_concepts() before that module is built; it isn't included
below on purpose, to avoid writing fabricated placeholder data.
"""

from typing import List, Dict, Optional
from datetime import datetime, timezone
from .db import get_connection
from ingestion.chunking import SemanticChunker
from ingestion.pdf_ingestion import extract_text_with_page_map, find_page_for_chunk
from ingestion.pdf_ingestion import extract_text_with_page_map,find_page_for_chunk


def resource_exists(file_hash: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT 1 FROM resources WHERE hash = ?",
        (file_hash,)
    )

    exists = cursor.fetchone() is not None

    conn.close()

    return exists


def save_resource(
    resource_id: str,
    name: str,
    resource_type: str,
    file_hash: str,
    path: str,
    summary: Optional[str] = None,
) -> None:
    """
    Insert or update a row in the resources table.
    summary can be None for now — fill it in later once the
    summarization step exists, via update_resource_summary().
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO resources (resource_id, name, type, hash, path, summary, created_at, modified_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(resource_id) DO UPDATE SET
            name=excluded.name,
            type=excluded.type,
            path=excluded.path,
            summary=COALESCE(excluded.summary, resources.summary),
            modified_at=excluded.modified_at
        """,
        (resource_id, name, resource_type, file_hash, path, summary, now, now),
    )
    conn.commit()
    conn.close()


def update_resource_summary(resource_id: str, summary: str) -> None:
    """Call this once your Gemma summarization step produces a summary."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE resources SET summary = ?, modified_at = ? WHERE resource_id = ?",
        (summary, datetime.now(timezone.utc).isoformat(), resource_id),
    )
    conn.commit()
    conn.close()


def save_chunks(
    chunks: List[Dict],
    resource_id: str,
    page_boundaries: Optional[List[Dict]] = None,
) -> int:
    """
    Insert chunk rows into the chunks table.

    Args:
        chunks: output of SemanticChunker.chunk() — each dict has
                chunk_id, chunk_text, start_index, end_index, etc.
        resource_id: parent resource id (must already exist in resources).
        page_boundaries: optional output of
                         pdf_ingestion.extract_text_with_page_map().
                         If provided, page_number is filled in per chunk
                         via find_page_for_chunk(). If None, page_number
                         is stored as NULL — fine for non-PDF sources
                         (notes, bookmarks) that have no page concept.

    Note: embedding_id is set to the same value as chunk_id — that's the
    exact id used when storing the vector in Chroma, so the two tables
    are always resolvable via one shared string, no separate id needed.

    Returns:
        Number of chunk rows written.
    """
    if not chunks:
        return 0

    page_lookup = None
    if page_boundaries:
        
        page_lookup = lambda start: find_page_for_chunk(start, page_boundaries)

    conn = get_connection()
    cursor = conn.cursor()

    rows = []
    for c in chunks:
        page_number = page_lookup(c["start_index"]) if page_lookup else None
        rows.append((
            c["chunk_id"],
            resource_id,
            c["chunk_text"],
            page_number,
            None,  # paragraph_number — not tracked yet, left NULL on purpose
            c["chunk_id"],  # embedding_id == chunk_id (same key used in Chroma)
        ))

    cursor.executemany(
        """
        INSERT OR REPLACE INTO chunks
            (chunk_id, resource_id, chunk_text, page_number, paragraph_number, embedding_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def log_interaction(query: str, resource_used: str) -> None:
    """Log a query + which resource answered it, for the interactions table
    (feeds the deterministic Evidence Engine: times-searched, last-used, etc)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO interactions (query, timestamp, resource_used) VALUES (?, ?, ?)",
        (query, datetime.now(timezone.utc).isoformat(), resource_used),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # end-to-end smoke test wiring the whole pipeline together
    

    # NOTE: replace with a real pdf path to actually run this
    pdf_path = "sample.pdf"

    import os
    if not os.path.exists(pdf_path):
        print(f"No {pdf_path} found — this is a wiring example, not a live test.")
    else:
        resource_id = "os_notes_pdf"
        full_text, page_boundaries = extract_text_with_page_map(pdf_path)

        save_resource(
            resource_id=resource_id,
            name="OS Notes",
            resource_type="pdf",
            file_hash="dummy_hash_replace_with_real_sha256",
            path=pdf_path,
        )

        chunker = SemanticChunker()
        chunks = chunker.chunk(full_text, resource_id=resource_id)
        n = save_chunks(chunks, resource_id=resource_id, page_boundaries=page_boundaries)
        print(f"Saved {n} chunks with page numbers mapped.")