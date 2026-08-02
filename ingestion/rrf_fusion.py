"""
rrf_fusion.py
Reciprocal Rank Fusion - merges multiple ranked ID lists into one ranking
without needing to reconcile different score scales (cosine vs bm25).
Only rank position matters, not the raw score value. Generic on purpose -
same function works for fusing resource IDs now and chunk IDs later
(build order step 3, hybrid chunk retrieval).
"""

from typing import List, Tuple
from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Args:
        ranked_lists: each is a list of ids, already ordered best-first.
                      An id missing from a list contributes nothing from
                      that list - it's not penalized to zero, just absent.
        k: standard RRF constant, 60 is the common default from the
           original paper - dampens the impact of rank 1 vs rank 2 so one
           list doesn't dominate just by ranking something slightly higher.

    Returns:
        [(id, fused_score), ...] sorted best-first.
    """
    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list, start=1):
            scores[item_id] += 1 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)