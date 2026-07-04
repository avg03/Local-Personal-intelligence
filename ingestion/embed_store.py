from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
 
 
def embed_and_store_chunks(
    chunks: List[Dict],
    resource_id: str,
    collection,
    encoder: Optional[SentenceTransformer] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> int:
    """
    Embed chunk texts and upsert them into an existing ChromaDB collection.
 
    Args:
        chunks: list of chunk dicts, each with at least
                chunk_id, chunk_text, start_index, end_index,
                chunk_length, token_estimate (as produced by SemanticChunker.chunk()).
        resource_id: id of the parent resource these chunks belong to
                     (stored in metadata for filtering/retrieval later).
        collection: your existing ChromaDB collection object
                    (e.g. the `collection` from your chroma init file —
                    pass it in directly, this function doesn't create one).
        encoder: a pre-loaded SentenceTransformer instance. Reuse the same
                 instance across calls where possible — loading the model
                 repeatedly wastes memory, which matters on constrained
                 hardware (laptop/mobile target for this project).
        embedding_model: used only if encoder is None.
        batch_size: batch size for encoding.
 
    Returns:
        Number of chunks embedded and stored.
    """
    if not chunks:
        return 0
 
    if encoder is None:
        encoder = SentenceTransformer(embedding_model)
 
    texts = [c["chunk_text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
 
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).tolist()
 
    metadatas = [
        {
            "resource_id": resource_id,
            "start_index": c.get("start_index", -1),
            "end_index": c.get("end_index", -1),
            "chunk_length": c.get("chunk_length", len(c["chunk_text"])),
            "token_estimate": c.get("token_estimate", len(c["chunk_text"]) // 4),
        }
        for c in chunks
    ]
 
    # upsert (not add) so re-ingesting the same resource overwrites
    # rather than raising a duplicate-id error or creating dupes
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
 
    return len(chunks)
 