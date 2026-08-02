"""
resource_resolution.py
Build order step 2: resolve which resources are relevant BEFORE touching
chunks. Fuses two signals: semantic (summary embedding similarity) and
lexical (FTS5 over name/summary) via RRF.
"""

from typing import List
from retrieval.resource_search import search_resources_by_summary
from database.fts_search import search_resources_fts
from rrf_fusion import reciprocal_rank_fusion


def resolve_resources(
    cleaned_query: str,
    query_embedding: List[float],
    summary_collection,
    top_k: int = 5,
    candidate_pool: int = 20,
) -> List[str]:
    """
    Args:
        cleaned_query: from query_processing.clean_query()
        query_embedding: from query_processing.process_query()
        summary_collection: Chroma collection with summary embeddings
        top_k: final number of resource_ids to return
        candidate_pool: how many results to pull from EACH signal before
                        fusing - wider than top_k so RRF has real signal
                        to work with, not just whatever already made top_k
                        in a single list.

    Returns:
        List of resource_ids, fused-ranked, best first.
    """
    semantic_results = search_resources_by_summary(query_embedding, summary_collection, top_k=candidate_pool)
    semantic_ranked_ids = [r["resource_id"] for r in semantic_results]

    lexical_results = search_resources_fts(cleaned_query, top_k=candidate_pool)
    lexical_ranked_ids = [r["resource_id"] for r in lexical_results]

    fused = reciprocal_rank_fusion([semantic_ranked_ids, lexical_ranked_ids])

    return [resource_id for resource_id, score in fused[:top_k]]


if __name__ == "__main__":
    import chromadb
    from encoder_config import encoder
    from retrieval.query_processing import process_query

    client = chromadb.PersistentClient(path="./chroma")
    summary_collection = client.get_or_create_collection(name="summary_embeddings")

    cleaned, embedding = process_query("Devcation submission requirements", encoder=encoder)
    top_resources = resolve_resources(cleaned, embedding, summary_collection, top_k=5)
    print(top_resources)