"""
chunk_search.py
Stage 2 of multi-stage retrieval: search chunk embeddings, but only
within the resource_ids narrowed down by resource_search.py's Stage 1.
Much cheaper than searching every chunk in the whole library.
"""

from typing import List, Dict
from resource_search import _check_cosine_metric  # same metric check, don't duplicate it


def search_chunks_by_resource_ids(
    query_embedding: List[float],
    resource_ids: List[str],
    collection,
    top_k: int = 10,
) -> List[Dict]:
    """
    Search the chunk embeddings collection, restricted to a set of
    resource_ids (from Stage 1 resource-level search).

    Args:
        query_embedding: embedding of the cleaned query — same encoder
                         used to build the chunk embeddings.
        resource_ids: list of resource_ids to search within (from
                      search_resources_by_summary()). If empty, returns [].
        collection: the ChromaDB collection storing chunk embeddings
                    (the one embed_and_store_chunks() writes to).
        top_k: how many chunks to return, across all matched resources.

    Returns:
        List of dicts, ranked by score (highest first):
        [{"chunk_id": ..., "resource_id": ..., "text": ..., "score": 0.88}]
    """
    if not query_embedding or not resource_ids:
        return []

    _check_cosine_metric(collection)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"resource_id": {"$in": resource_ids}},
        include=["metadatas", "distances", "documents"],
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]

    ranked = []
    for chunk_id, distance, metadata, document in zip(ids, distances, metadatas, documents):
        score = 1 - distance  # valid only if collection uses cosine space — see warning above
        ranked.append({
            "chunk_id": chunk_id,
            "resource_id": metadata.get("resource_id"),
            "text": document,
            "score": score,
        })

    return ranked


if __name__ == "__main__":
    import chromadb
    from ingestion.encoder_config import encoder
    from query_processing import process_query
    from resource_search import search_resources_by_summary

    client = chromadb.PersistentClient(path="./chroma")
    chunk_collection = client.get_or_create_collection(name="student_memory")
    summary_collection = client.get_or_create_collection(name="summary_embeddings")

    _, query_embedding = process_query("how do you avoid deadlocks", encoder=encoder)

    top_resources = search_resources_by_summary(query_embedding, summary_collection, top_k=5)
    resource_ids = [r["resource_id"] for r in top_resources]

    top_chunks = search_chunks_by_resource_ids(query_embedding, resource_ids, chunk_collection, top_k=10)
    for c in top_chunks:
        print(f"{c['chunk_id']}  score={c['score']:.3f}  {c['text'][:60]}")