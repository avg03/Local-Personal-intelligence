"""
hybrid_chunk_retrieval.py
Build order: after resource resolution narrows WHICH resources, this
narrows WHICH chunks within them - fusing semantic (chunk_search.py) and
lexical (fts_search.py) results via RRF. Output feeds the reranker next.
"""

from typing import List, Dict
from chunk_search import search_chunks_by_resource_ids
from database.fts_search import search_chunks_fts
from ingestion.rrf_fusion import reciprocal_rank_fusion


def hybrid_chunk_retrieval(
    cleaned_query: str,
    query_embedding: List[float],
    resource_ids: List[str],
    chunk_collection,
    top_k: int = 20,
    candidate_pool: int = 30,
) -> List[Dict]:
    """
    Returns fused-ranked chunks, best first:
    [{"chunk_id": ..., "resource_id": ..., "text": ..., "score": ...}]
    score here is the RRF fused score, not a raw similarity/bm25 value -
    don't compare it against scores from earlier stages, it's a different scale.
    """
    semantic_results = search_chunks_by_resource_ids(query_embedding, resource_ids, chunk_collection, top_k=candidate_pool)
    lexical_results = search_chunks_fts(cleaned_query, resource_ids=resource_ids, top_k=candidate_pool)

    # RRF only returns (id, score) - need this to reattach text/resource_id after fusion
    chunk_lookup = {}
    for r in semantic_results + lexical_results:
        chunk_lookup[r["chunk_id"]] = r

    semantic_ids = [r["chunk_id"] for r in semantic_results]
    lexical_ids = [r["chunk_id"] for r in lexical_results]
    fused = reciprocal_rank_fusion([semantic_ids, lexical_ids])

    results = []
    for chunk_id, fused_score in fused[:top_k]:
        base = chunk_lookup[chunk_id]
        results.append({
            "chunk_id": chunk_id,
            "resource_id": base["resource_id"],
            "text": base["text"],
            "score": fused_score,
        })
    return results


if __name__ == "__main__":
    import chromadb
    from ingestion.encoder_config import encoder
    from query_processing import process_query
    from ingestion.resource_resolution import resolve_resources

    client = chromadb.PersistentClient(path="./chroma")
    chunk_collection = client.get_or_create_collection(name="student_memory")
    summary_collection = client.get_or_create_collection(name="summary_embeddings")

    cleaned, embedding = process_query("Devcation submission requirements", encoder=encoder)
    resource_ids = resolve_resources(cleaned, embedding, summary_collection, top_k=5)
    candidates = hybrid_chunk_retrieval(cleaned, embedding, resource_ids, chunk_collection)

    for c in candidates:
        print(f"{c['chunk_id']}  score={c['score']:.4f}  {c['text'][:60]}")