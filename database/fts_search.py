"""
fts_search.py
Lexical chunk search via SQLite FTS5 — catches identifiers/exact terms
("Devcation", "CS101") that semantic search can miss. Same output shape
as chunk_search.py so the two can be merged with RRF later (build order
step 2).
"""

from typing import List, Dict, Optional
from database.db import get_connection


def _build_fts_query(cleaned_query: str) -> str:
    """
    FTS5 has its own query syntax (AND/OR/NOT, *, -, parens) — a raw user
    query containing any of those characters can break the MATCH clause
    or silently change meaning. Quoting each term treats it as a literal
    string, sidestepping that. Doubling internal quotes escapes them per
    FTS5's own rule for quoted strings.
    """
    terms = cleaned_query.split()
    quoted_terms = [f'"{t.replace(chr(34), chr(34)*2)}"' for t in terms if t]
    return " OR ".join(quoted_terms)


def search_chunks_fts(
    cleaned_query: str,
    resource_ids: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Dict]:
    """
    Search chunks_fts for lexical matches.

    Args:
        cleaned_query: output of query_processing.clean_query() — don't
                       pass raw unprocessed input here.
        resource_ids: optional narrowing, same role as in chunk_search.py's
                      Stage 2 — if given, only searches within these resources.
        top_k: max results.

    Returns:
        [{"chunk_id": ..., "resource_id": ..., "text": ..., "score": ...}]
        score is normalized so higher = more relevant, matching the
        convention used everywhere else in the pipeline. Raw SQLite bm25()
        returns MORE NEGATIVE for better matches — the opposite of every
        other score in this codebase — so it's negated here to keep that
        convention consistent instead of surprising whoever reads this later.
    """
    if not cleaned_query:
        return []

    fts_query = _build_fts_query(cleaned_query)
    if not fts_query:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    if resource_ids:
        placeholders = ",".join("?" for _ in resource_ids)
        sql = f"""
            SELECT chunk_id, resource_id, chunk_text, bm25(chunks_fts) as rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ? AND resource_id IN ({placeholders})
            ORDER BY rank
            LIMIT ?
        """
        params = [fts_query, *resource_ids, top_k]
    else:
        sql = """
            SELECT chunk_id, resource_id, chunk_text, bm25(chunks_fts) as rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        params = [fts_query, top_k]

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        {"chunk_id": chunk_id, "resource_id": resource_id, "text": text, "score": -rank}
        for chunk_id, resource_id, text, rank in rows
    ]


def search_resources_fts(
    cleaned_query: str,
    top_k: int = 10,
) -> List[Dict]:
    """
    Search resources_fts for lexical matches over name + summary.

    Returns:
        [{"resource_id": ..., "name": ..., "summary": ..., "score": ...}]
    """
    if not cleaned_query:
        return []

    fts_query = _build_fts_query(cleaned_query)
    if not fts_query:
        return []

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT resource_id, name, summary, bm25(resources_fts) as rank
        FROM resources_fts
        WHERE resources_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (fts_query, top_k),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {"resource_id": resource_id, "name": name, "summary": summary, "score": -rank}
        for resource_id, name, summary, rank in rows
    ]


if __name__ == "__main__":
    from retrieval.query_processing import clean_query

    cleaned = clean_query("Devcation submission requirements")
    results = search_chunks_fts(cleaned, top_k=5)
    for r in results:
        print(f"{r['chunk_id']}  score={r['score']:.3f}  {r['text'][:60]}")