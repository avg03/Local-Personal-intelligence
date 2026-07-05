"""
knowledge_structure.py
Builds a KnowledgeContext from an Evidence Object (evidence_builder.py output).

This is the Connection Engine, built the way it was flagged it should be
during the architecture review: 100% deterministic co-occurrence math,
no LLM inference of relationships. Gemma only interprets this structure
later (in the Reflection stage) — it never generates it.
"""

from typing import List, Dict
from collections import Counter, defaultdict


def _resource_concepts(evidence: List[Dict]) -> List[Dict]:
    """Internal: (resource_name, deduped concept list) per evidence entry,
    in the order evidence was given (already sorted by retrieval_score)."""
    out = []
    for e in evidence:
        concepts = list(dict.fromkeys(e.get("concepts", [])))  # dedupe, preserve order
        out.append({"resource_name": e.get("resource_name"), "concepts": concepts})
    return out


def compute_concept_frequency(evidence: List[Dict]) -> Dict[str, int]:
    """
    Count how many distinct resources (in this evidence set) mention each
    concept. Not chunk count, not raw mentions — resource-level presence,
    since that's the meaningful cross-resource signal for "how established
    is this concept across what the student has studied."
    """
    freq = Counter()
    for r in _resource_concepts(evidence):
        for concept in r["concepts"]:
            freq[concept] += 1
    return dict(freq)


def extract_main_concepts(evidence: List[Dict]) -> List[str]:
    """
    Concepts across the evidence set, ranked by frequency (most
    cross-resource concepts first), tie-broken by order of first
    appearance for determinism.
    """
    freq = compute_concept_frequency(evidence)

    first_seen_order = []
    seen = set()
    for r in _resource_concepts(evidence):
        for concept in r["concepts"]:
            if concept not in seen:
                seen.add(concept)
                first_seen_order.append(concept)

    return sorted(first_seen_order, key=lambda c: (-freq[c], first_seen_order.index(c)))


def build_concept_relations(evidence: List[Dict]) -> Dict[str, List[str]]:
    """
    Co-occurrence graph: for each concept, which other concepts appear
    alongside it in the same resource(s). Pure counting — two concepts
    in the same resource's concept list are considered related; strength
    ranked by how many resources they co-occur in.
    """
    co_occurrence = defaultdict(Counter)

    for r in _resource_concepts(evidence):
        concepts = r["concepts"]
        for i, a in enumerate(concepts):
            for b in concepts:
                if a != b:
                    co_occurrence[a][b] += 1

    related = {}
    for concept, counter in co_occurrence.items():
        # sort by co-occurrence strength desc, then alphabetically for stable ties
        related[concept] = sorted(counter.keys(), key=lambda c: (-counter[c], c))

    return related


def find_related_resources(evidence: List[Dict]) -> List[str]:
    """
    Resource names present in this evidence set, in retrieval-score order
    (evidence is already sorted that way coming out of build_evidence()).
    "Related" here means "the resources this query's evidence already
    spans" — not a broader library-wide search, evidence is the scope.
    """
    seen = set()
    ordered_names = []
    for e in evidence:
        name = e.get("resource_name")
        if name and name not in seen:
            seen.add(name)
            ordered_names.append(name)
    return ordered_names


def build_knowledge_context(evidence: List[Dict]) -> Dict:
    """
    Combine all four functions into the full KnowledgeContext structure,
    ready to hand to Gemma's Reflection stage alongside the Evidence Object.
    """
    return {
        "main_concepts": extract_main_concepts(evidence),
        "concept_frequency": compute_concept_frequency(evidence),
        "related_concepts": build_concept_relations(evidence),
        "related_resources": find_related_resources(evidence),
    }


if __name__ == "__main__":
    import json

    sample_evidence = [
        {
            "resource_id": "os_notes_pdf", "resource_name": "OS.pdf",
            "summary": "Covers deadlocks, semaphores, and mutexes.",
            "concepts": ["Deadlock", "Semaphore", "Mutex"],
            "retrieved_chunks": [], "retrieval_score": 0.91,
        },
        {
            "resource_id": "assignment3", "resource_name": "Assignment3.pdf",
            "summary": "Deadlock avoidance assignment using Banker's Algorithm.",
            "concepts": ["Deadlock", "Banker's Algorithm"],
            "retrieved_chunks": [], "retrieval_score": 0.75,
        },
        {
            "resource_id": "networking_notes", "resource_name": "Networking.pdf",
            "summary": None, "concepts": [],  # no concepts extracted yet for this one
            "retrieved_chunks": [], "retrieval_score": 0.40,
        },
    ]

    context = build_knowledge_context(sample_evidence)
    print(json.dumps(context, indent=2))