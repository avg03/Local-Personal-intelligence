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

#will store summary embedddings in chormaDB also .
def embed_summary(summary_text: str, encoder: Optional[SentenceTransformer] = None, embedding_model: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Embed a summary text using the provided encoder or a new SentenceTransformer instance.

    Args:
        summary_text: The summary text to embed.
        encoder: A pre-loaded SentenceTransformer instance. Reuse the same
                 instance across calls where possible.
        embedding_model: Used only if encoder is None."""

    if encoder is None:
        encoder = SentenceTransformer(embedding_model)

    embedding = encoder.encode(summary_text, convert_to_numpy=True).tolist()
    return embedding 

#save the summary embeddings in the summary_embeddings collection in ChromaDB
def store_summary_embedding(summary_text: str, resource_id: str, collection, encoder: Optional[SentenceTransformer] = None, embedding_model: str = "all-MiniLM-L6-v2") -> None:
    """
    Embed a summary text and store it in the provided ChromaDB collection.

    Args:
        summary_text: The summary text to embed and store.
        resource_id: ID of the parent resource this summary belongs to.
        collection: Your existing ChromaDB collection object for storing summary embeddings.
        encoder: A pre-loaded SentenceTransformer instance. Reuse the same
                 instance across calls where possible.
        embedding_model: Used only if encoder is None.
    """
    embedding = embed_summary(summary_text, encoder=encoder, embedding_model=embedding_model)

    # Create a unique ID for the summary embedding based on the resource_id
    summary_id = f"summary_{resource_id}"

    # Upsert the summary embedding into the collection
    collection.upsert(
        ids=[summary_id],
        embeddings=[embedding],
        documents=[summary_text],
        metadatas=[{"resource_id": resource_id}],
    )