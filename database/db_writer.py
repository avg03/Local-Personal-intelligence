from typing import List, Dict, Optional
from datetime import datetime, timezone
from db import get_connection


def resource_exists(file_hash: str) -> bool:
    """Check by content hash, not resource_id - catches re-ingestion of
    the same file even under a different generated id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM resources WHERE hash = ? LIMIT 1", (file_hash,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def save_resource(resource_id, name, resource_type, file_hash, path, summary=None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO resources (resource_id, name, type, hash, path, summary, created_at, modified_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(resource_id) DO UPDATE SET
            name=excluded.name, type=excluded.type, path=excluded.path,
            summary=COALESCE(excluded.summary, resources.summary), modified_at=excluded.modified_at
        """,
        (resource_id, name, resource_type, file_hash, path, summary, now, now),
    )
    conn.commit()
    conn.close()


def update_resource_summary(resource_id: str, summary: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE resources SET summary = ?, modified_at = ? WHERE resource_id = ?",
        (summary, datetime.now(timezone.utc).isoformat(), resource_id),
    )
    conn.commit()
    conn.close()


def save_chunks(chunks: List[Dict], resource_id: str, page_boundaries: Optional[List[Dict]] = None) -> int:
    if not chunks:
        return 0

    page_lookup = None
    if page_boundaries:
        from ingestion.pdf_ingestion import find_page_for_chunk
        page_lookup = lambda start: find_page_for_chunk(start, page_boundaries)

    conn = get_connection()
    cursor = conn.cursor()
    rows = []
    for c in chunks:
        page_number = page_lookup(c["start_index"]) if page_lookup else None
        rows.append((c["chunk_id"], resource_id, c["chunk_text"], page_number, None, c["chunk_id"]))

    cursor.executemany(
        """INSERT OR REPLACE INTO chunks
           (chunk_id, resource_id, chunk_text, page_number, paragraph_number, embedding_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def log_interaction(query: str, resource_used: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO interactions (query, timestamp, resource_used) VALUES (?, ?, ?)",
        (query, datetime.now(timezone.utc).isoformat(), resource_used),
    )
    conn.commit()
    conn.close()