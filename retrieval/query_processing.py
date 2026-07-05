"""
query_processing.py
Cleans a raw user query and embeds it using the same shared encoder
used for chunk embeddings — must match, or query and stored vectors
won't live in the same space (see earlier embedding-space mismatch note).
"""

import re
from typing import Tuple, List, Optional
from sentence_transformers import SentenceTransformer


def clean_query(query: str) -> str:
    """
    Light cleaning for a search query — NOT the same as clean_resume_text.
    Queries are short and typed by a human right now, so this only strips
    noise, it doesn't try to fix OCR artifacts or reflow paragraphs.
    """
    if not query:
        return ""

    cleaned = query.strip()

    # remove control/non-printable characters
    cleaned = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", " ", cleaned)

    # collapse repeated whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def process_query(
    query: str,
    encoder: Optional[SentenceTransformer] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Tuple[str, List[float]]:
    """
    Clean a query and embed it.

    Args:
        query: raw user input.
        encoder: shared SentenceTransformer instance — pass the same one
                 used in embed_and_store.py / semantic_chunker.py. If None,
                 a new one is loaded (avoid this in production paths;
                 reuse the shared instance instead, see encoder_config.py).
        embedding_model: only used if encoder is None.

    Returns:
        (cleaned_query, embedding) — embedding is a plain list of floats,
        ready to pass straight into collection.query(query_embeddings=...).
    """
    cleaned_query = clean_query(query)

    if not cleaned_query:
        return "", []

    if encoder is None:
        encoder = SentenceTransformer(embedding_model)

    embedding = encoder.encode(cleaned_query, convert_to_numpy=True).tolist()

    return cleaned_query, embedding


if __name__ == "__main__":
    from ingestion.encoder_config import encoder

    raw_query = "  how does   the Banker's Algorithm avoid deadlock??  \n"
    cleaned, embedding = process_query(raw_query, encoder=encoder)
    print("Cleaned:", repr(cleaned))
    print("Embedding length:", len(embedding))