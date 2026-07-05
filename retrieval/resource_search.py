"""
resource_search.py
Stage 1 of multi-stage retrieval: search the *summary* embeddings
collection to narrow down which resources are relevant, before doing
the more expensive chunk-level search only inside those resources.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def _check_cosine_metric(collection) -> None:
    """
    Chroma defaults to L2 distance, not cosine, unless the collection was
    created with metadata={"hnsw:space": "cosine"}. If that wasn't set on
    your summary_embeddings collection, "top-k by similarity" below is
    silently ranking by the wrong metric. Warn loudly rather than fail
    quietly on a wrong ranking.
    """
    metadata = getattr(collection, "metadata", None) or {}
    space = metadata.get("hnsw:space")
    if space != "cosine":
        logger.warning(
            "Collection '%s' has hnsw:space=%s (not 'cosine'). "
            "Similarity scores below will be computed as if cosine, but "
            "the underlying index was built for a different metric — "
            "recreate the collection with metadata={'hnsw:space': 'cosine'} "
            "to get correct rankings.",
            getattr(collection, "name", "?"), space,
        )


def search_resources_by_summary(
    query_embedding: List[float],
    collection,
    top_k: int = 5,
) -> List[Dict]:
    """
    Search the summary_embeddings collection for the top-k resources whose
    summary is most similar to the query embedding.

    Args:
        query_embedding: embedding of the (cleaned) user query — from
                         query_processing.process_query(), same encoder
                         used to build the summary embeddings.
        collection: the ChromaDB collection storing summary embeddings
                    (created via store_summary_embedding()).
        top_k: how many resources to return.

    Returns:
        List of dicts, ranked by similarity (highest first):
        {resource_id, summary_id, summary_text, similarity}
        `similarity` is in [-1, 1] (cosine similarity), higher = closer.
    """
    if not query_embedding:
        return []

    _check_cosine_metric(collection)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]

    ranked = []
    for summary_id, distance, metadata, document in zip(ids, distances, metadatas, documents):
        # cosine distance in Chroma = 1 - cosine_similarity, when the
        # collection is actually configured for cosine space (see check above)
        similarity = 1 - distance
        ranked.append({
            "resource_id": metadata.get("resource_id"),
            "summary_id": summary_id,
            "summary_text": document,
            "similarity": similarity,
        })

    return ranked


if __name__ == "__main__":
    import chromadb
    from ingestion.encoder_config import encoder
    from query_processing import process_query
    from ingestion.embed_store import store_summary_embedding  # wherever embed_summary/store_summary_embedding live

    client = chromadb.PersistentClient(path="./chroma")
    summary_collection = client.get_or_create_collection(
        name="summary_embeddings",
        metadata={"hnsw:space": "cosine"},  # must set this at creation time
    )

    store_summary_embedding(
        "Covers deadlocks, banker's algorithm, and process scheduling.",
        resource_id="os_notes_pdf",
        collection=summary_collection,
        encoder=encoder,
    )

    _, query_embedding = process_query("how do you avoid deadlocks", encoder=encoder)
    top_resources = search_resources_by_summary(query_embedding, summary_collection, top_k=3)
    for r in top_resources:
        print(f"{r['resource_id']}  (similarity={r['similarity']:.3f})  {r['summary_text'][:60]}")