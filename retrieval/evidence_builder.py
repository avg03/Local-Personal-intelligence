from typing import List, Dict
from collections import defaultdict
from database.db import get_connection


def _fetch_resource_metadata(resource_ids: List[str]) -> Dict[str, Dict]:
    if not resource_ids:
        return {}
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in resource_ids)
    # path added here - reasoning_agent_node needs it for citing sources
    cursor.execute(
        f"SELECT resource_id, name, summary, path FROM resources WHERE resource_id IN ({placeholders})",
        resource_ids,
    )
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: {"name": row[1], "summary": row[2], "path": row[3]} for row in rows}


def _fetch_concepts(resource_ids: List[str]) -> Dict[str, List[str]]:
    if not resource_ids:
        return {}
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in resource_ids)
    cursor.execute(
        f"SELECT resource_id, concept FROM concepts WHERE resource_id IN ({placeholders}) ORDER BY confidence DESC",
        resource_ids,
    )
    rows = cursor.fetchall()
    conn.close()
    grouped = defaultdict(list)
    for resource_id, concept in rows:
        grouped[resource_id].append(concept)
    return dict(grouped)


def _fetch_chunk_pages(chunk_ids: List[str]) -> Dict[str, int]:
    if not chunk_ids:
        return {}
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in chunk_ids)
    cursor.execute(f"SELECT chunk_id, page_number FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def build_evidence(chunks: List[Dict], aggregation: str = "max") -> List[Dict]:
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
            "resource_path": meta.get("path"),  # new - reasoning_agent_node reads this for citations
            "summary": meta.get("summary"),
            "concepts": concepts_by_resource.get(resource_id, []),
            "retrieved_chunks": [
                {"page": page_by_chunk.get(c["chunk_id"]), "text": c["text"]}
                for c in resource_chunks
            ],
            "retrieval_score": round(retrieval_score, 4),
        })

    evidence.sort(key=lambda e: e["retrieval_score"], reverse=True)
    return evidence