"""
relevance_gate.py
Dynamic filtering, not fixed top-K. Only chunks that clear the threshold
survive - could be 0 (NO MATCH), 2, or the full max_keep. This is the
gate that stops topically-similar-but-not-answer-bearing chunks from
reaching the reasoning agent.
"""

from typing import List, Dict

# Starting point, NOT a measured value - flagged the same way as
# LEARNING_STATE_THRESHOLDS in reflection_context.py. Needs a real
# labeled eval set (even 30-50 queries) before trusting this number.
RELEVANCE_THRESHOLD = 0.5


def apply_relevance_gate(
    reranked_chunks: List[Dict],
    threshold: float = RELEVANCE_THRESHOLD,
    max_keep: int = 5,
) -> List[Dict]:
    """
    Args:
        reranked_chunks: output of rerank_chunks(), already sorted best first.
        threshold: minimum rerank_score to survive. Chunks below this are
                   dropped even if they were in the top-K candidate pool -
                   "topically related" isn't the same as "answer-bearing".
        max_keep: hard cap even if more chunks clear the threshold.

    Returns:
        Filtered list, best first. Empty list means NO MATCH - caller
        should surface that explicitly, not silently pass an empty
        evidence set downstream as if nothing went wrong.
    """
    kept = [c for c in reranked_chunks if c["rerank_score"] >= threshold]
    return kept[:max_keep]