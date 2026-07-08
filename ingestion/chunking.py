"""
semantic_chunker.py
Similarity-threshold semantic chunking (Greg Kamradt's approach), refined:
- sentence-index ranges tracked through merge/split (no stale group->chunk mapping)
- exact character offsets sliced from source text (no text.find() guessing)
- percentile-based adaptive threshold, with fixed-threshold fallback
"""

import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from dotenv import load_dotenv
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer


def _ensure_nltk_data():
    """Download punkt tokenizer data once, quietly, if missing."""
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass  # older nltk versions may not have punkt_tab; punkt alone is enough


_ensure_nltk_data()
from nltk.tokenize import sent_tokenize  # noqa: E402


class SemanticChunker:
    """
    Semantic chunker using similarity threshold between consecutive
    sentence groups to detect topic boundaries.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,     # used only if use_percentile_threshold=False
        use_percentile_threshold: bool = True, # Kamradt's actual method: adaptive per-document
        percentile: float = 95.0,              # higher = fewer, bigger chunks
        sentences_per_group: int = 3,
        overlap_sentences: int = 1,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        load_dotenv()
        self.encoder = SentenceTransformer(embedding_model, token=os.getenv("HUGGING_FACE_TOKEN"))
        self.similarity_threshold = similarity_threshold
        self.use_percentile_threshold = use_percentile_threshold
        self.percentile = percentile
        self.sentences_per_group = sentences_per_group
        self.overlap_sentences = overlap_sentences
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    # ---------- sentence handling ----------

    def _sentences_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split into sentences and find each one's real (start, end) offset
        in the source text via a moving cursor — avoids grabbing the wrong
        occurrence when a phrase repeats (headers, boilerplate, etc).
        """
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        spans = []
        cursor = 0
        for sent in sentences:
            idx = text.find(sent, cursor)
            if idx == -1:
                idx = cursor  # fallback: shouldn't normally happen
            start, end = idx, idx + len(sent)
            spans.append((sent, start, end))
            cursor = end
        return spans

    def _make_groups(
        self, spans: List[Tuple[str, int, int]]
    ) -> List[Tuple[int, int]]:
        """Return list of (start_sent_idx, end_sent_idx_exclusive) per group."""
        groups = []
        n = len(spans)
        for i in range(0, n, self.sentences_per_group):
            groups.append((i, min(i + self.sentences_per_group, n)))
        return groups

    def _group_text(self, spans, group_range: Tuple[int, int]) -> str:
        start, end = group_range
        return " ".join(spans[i][0] for i in range(start, end))

    # ---------- similarity ----------

    def _calculate_similarities(self, group_texts: List[str]) -> List[float]:
        if len(group_texts) < 2:
            return []
        embeddings = self.encoder.encode(group_texts, show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.clip(norms, 1e-8, None)
        sims = [float(np.dot(normed[i], normed[i + 1])) for i in range(len(normed) - 1)]
        return sims

    def _resolve_threshold(self, similarities: List[float]) -> float:
        if not similarities:
            return self.similarity_threshold
        if not self.use_percentile_threshold:
            return self.similarity_threshold
        # Kamradt's method: split where the *distance* spikes — take the
        # Nth percentile of distances as the cut line, converted back to similarity.
        distances = [1 - s for s in similarities]
        cutoff_distance = float(np.percentile(distances, self.percentile))
        return 1 - cutoff_distance

    # ---------- sentence-range chunk building (bug fix lives here) ----------

    def _groups_to_sentence_ranges(
        self, groups: List[Tuple[int, int]], split_points: List[int]
    ) -> List[Tuple[int, int]]:
        """Convert group-level split points directly into sentence-index
        ranges per chunk — this is the single source of truth carried
        through merge/split/overlap, so nothing can go stale."""
        ranges = []
        prev = 0
        for point in split_points:
            ranges.append((groups[prev][0], groups[point - 1][1]))
            prev = point
        if prev < len(groups):
            ranges.append((groups[prev][0], groups[-1][1]))
        return ranges

    def _apply_size_constraints(
        self, spans: List[Tuple[str, int, int]], ranges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Merge tiny ranges into neighbors, split oversized ranges at
        sentence boundaries. Operates purely on sentence-index ranges."""

        def range_len(r):
            start, end = r
            return spans[end - 1][2] - spans[start][1]  # char span length

        # merge tiny chunks forward/backward
        merged: List[Tuple[int, int]] = []
        for r in ranges:
            if range_len(r) < self.min_chunk_size and merged:
                prev_start, _ = merged[-1]
                merged[-1] = (prev_start, r[1])
            else:
                merged.append(r)

        # split oversized chunks at sentence boundaries
        final: List[Tuple[int, int]] = []
        for start, end in merged:
            seg_start = start
            acc_len = 0
            for i in range(start, end):
                sent_len = spans[i][2] - spans[i][1]
                if acc_len + sent_len > self.max_chunk_size and i > seg_start:
                    final.append((seg_start, i))
                    seg_start = i
                    acc_len = sent_len
                else:
                    acc_len += sent_len
            final.append((seg_start, end))

        return final

    def _apply_overlap(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Extend each chunk's start backward by N sentences for context
        continuity. Pure index math — always valid, nothing can go stale."""
        if self.overlap_sentences <= 0:
            return ranges
        overlapped = []
        for i, (start, end) in enumerate(ranges):
            new_start = max(0, start - self.overlap_sentences) if i > 0 else start
            overlapped.append((new_start, end))
        return overlapped

    # ---------- public API ----------

    def chunk(self, text: str, resource_id: Optional[str] = None) -> List[Dict]:
        """
        Chunk text using similarity-threshold boundary detection.

        Args:
            text: source text (already cleaned, e.g. via clean_resume_text).
            resource_id: optional id to prefix chunk_id with, for tying
                         chunks back to a resource row in SQLite.

        Returns:
            List of dicts: chunk_id, chunk_text, start_index, end_index,
            chunk_length, token_estimate.
        """
        spans = self._sentences_with_offsets(text)
        if not spans:
            return []

        groups = self._make_groups(spans)
        group_texts = [self._group_text(spans, g) for g in groups]
        similarities = self._calculate_similarities(group_texts)
        threshold = self._resolve_threshold(similarities)

        split_points = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

        ranges = self._groups_to_sentence_ranges(groups, split_points)
        ranges = self._apply_size_constraints(spans, ranges)
        ranges = self._apply_overlap(ranges)

        results = []
        for i, (start, end) in enumerate(ranges):
            char_start = spans[start][1]
            char_end = spans[end - 1][2]
            chunk_text = text[char_start:char_end]  # exact slice, preserves original spacing
            prefix = f"{resource_id}_chunk_{i:04d}" if resource_id else f"chunk_{i:04d}"
            results.append({
                "chunk_id": prefix,
                "chunk_text": chunk_text,
                "start_index": char_start,
                "end_index": char_end,
                "chunk_length": len(chunk_text),
                "token_estimate": len(chunk_text) // 4,
            })
        return results




