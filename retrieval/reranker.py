"""
reranker.py
Cross-encoder reranking - scores (query, chunk_text) pairs directly,
one forward pass per pair. Input is the candidate pool from
hybrid_chunk_retrieval.py, output feeds relevance_gate.py next.
"""

import math
from typing import List, Dict
from sentence_transformers import CrossEncoder


def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    # ~80MB, CPU-fine, no GPU needed - load once, reuse like encoder/llm elsewhere
    return CrossEncoder(model_name)


def rerank_chunks(query: str, candidates: List[Dict], reranker: CrossEncoder) -> List[Dict]:
    """
    Args:
        query: cleaned user query.
        candidates: output of hybrid_chunk_retrieval() - list of
                    {chunk_id, resource_id, text, score}.
        reranker: loaded via load_reranker(), reuse the same instance.

    Returns:
        Same dicts with "rerank_score" added (0-1, sigmoid of the raw
        logit - raw scores are unbounded and not meaningfully
        thresholdable on their own), sorted best first.
    """
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    raw_scores = reranker.predict(pairs)

    reranked = []
    for c, raw in zip(candidates, raw_scores):
        prob = 1 / (1 + math.exp(-raw))
        reranked.append({**c, "rerank_score": prob})

    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return reranked