"""
evidence_builder.py
Combines chunk search results (from chunk_search.py) with resource
metadata and concepts (from SQLite) into structured Evidence Objects
ready to hand to Gemma.

Note: `concepts` will come back as an empty list for every resource
until the concept-extraction module exists (flagged as not-built in
db_writer.py earlier) — this isn't a bug here, there's just nothing in
the concepts table yet to join against.
"""

from typing import List, Dict
from collections import defaultdict
from database.db import get_connection


def _fetch_resource_metadata(resource_ids: List[str]) -> Dict[str, Dict]:
    """Batch-fetch name + summary for a list of resource_ids in one query."""
    if not resource_ids:
        return {}
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in resource_ids)
    cursor.execute(
        f"SELECT resource_id, name, summary FROM resources WHERE resource_id IN ({placeholders})",
        resource_ids,
    )
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: {"name": row[1], "summary": row[2]} for row in rows}


def _fetch_concepts(resource_ids: List[str]) -> Dict[str, List[str]]:
    """Batch-fetch concepts per resource_id. Empty list if none exist yet
    (expected until the concept-extraction module is built)."""
    if not resource_ids:
        return {}
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in resource_ids)
    cursor.execute(
        f"SELECT resource_id, concept FROM concepts WHERE resource_id IN ({placeholders}) "
        f"ORDER BY confidence DESC",
        resource_ids,
    )
    rows = cursor.fetchall()
    conn.close()
    grouped = defaultdict(list)
    for resource_id, concept in rows:
        grouped[resource_id].append(concept)
    return dict(grouped)


def _fetch_chunk_pages(chunk_ids: List[str]) -> Dict[str, int]:
    """Batch-fetch page_number per chunk_id. Chroma's metadata doesn't
    store page_number — only SQLite's chunks table does — so this join
    is necessary to put "page" in the output at all."""
    if not chunk_ids:
        return {}
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in chunk_ids)
    cursor.execute(
        f"SELECT chunk_id, page_number FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    )
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def build_evidence(chunks: List[Dict], aggregation: str = "max") -> List[Dict]:
    """
    Combine chunk search results with resource metadata + concepts into
    Evidence Objects, grouped by resource, ready for Gemma.

    Args:
        chunks: output of chunk_search.search_chunks_by_resource_ids() —
                list of {chunk_id, resource_id, text, score}.
        aggregation: how to roll up per-chunk scores into a single
                     resource-level retrieval_score. "max" = strongest
                     single match in that resource (default — surfaces
                     resources with one very relevant chunk even if the
                     rest is unrelated). "mean" = average relevance
                     across all matched chunks in that resource.

    Returns:
        List of Evidence Objects, sorted by retrieval_score descending:
        [{
            "resource_id": ..., "resource_name": ..., "summary": ...,
            "concepts": [...],
            "retrieved_chunks": [{"page": ..., "text": ...}, ...],
            "retrieval_score": ...
        }]
    """
    if not chunks:
        return []

    if aggregation not in ("max", "mean"):
        raise ValueError("aggregation must be 'max' or 'mean'")

    grouped_chunks = defaultdict(list)
    for c in chunks:
        grouped_chunks[c["resource_id"]].append(c)

    resource_ids = list(grouped_chunks.keys())
    chunk_ids = [c["chunk_id"] for c in chunks]

    resource_meta = _fetch_resource_metadata(resource_ids)
    concepts_by_resource = _fetch_concepts(resource_ids)
    page_by_chunk = _fetch_chunk_pages(chunk_ids)

    evidence = []
    for resource_id, resource_chunks in grouped_chunks.items():
        meta = resource_meta.get(resource_id, {})
        scores = [c["score"] for c in resource_chunks]
        retrieval_score = max(scores) if aggregation == "max" else sum(scores) / len(scores)

        evidence.append({
            "resource_id": resource_id,
            "resource_name": meta.get("name", "Unknown resource"),
            "summary": meta.get("summary"),  # may be None if not summarized yet
            "concepts": concepts_by_resource.get(resource_id, []),  # empty until concept extraction is built
            "retrieved_chunks": [
                {"page": page_by_chunk.get(c["chunk_id"]), "text": c["text"]}
                for c in resource_chunks
            ],
            "retrieval_score": round(retrieval_score, 4),
        })

    evidence.sort(key=lambda e: e["retrieval_score"], reverse=True)
    return evidence


if __name__ == "__main__":
    from ingestion.encoder_config import encoder
    from query_processing import process_query
    from resource_search import search_resources_by_summary
    from chunk_search import search_chunks_by_resource_ids
    import chromadb

    client = chromadb.PersistentClient(path="./chroma")
    chunk_collection = client.get_or_create_collection(name="student_memory")
    summary_collection = client.get_or_create_collection(name="summary_embeddings")

    _, query_embedding = process_query("how do you avoid deadlocks", encoder=encoder)
    top_resources = search_resources_by_summary(query_embedding, summary_collection, top_k=5)
    resource_ids = [r["resource_id"] for r in top_resources]
    top_chunks = search_chunks_by_resource_ids(query_embedding, resource_ids, chunk_collection, top_k=10)

    evidence = build_evidence(top_chunks)
    import json
    print(json.dumps(evidence, indent=2))