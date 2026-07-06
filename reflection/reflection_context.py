"""
reflection_context.py
The deterministic Evidence Engine half of the Reflection Engine.
Takes query + Evidence Object + KnowledgeContext and computes structured
facts about the retrieval — no LLM calls anywhere in this file. Gemma's
job (elsewhere, not here) is to interpret this structure into natural
language — this file only produces the facts it interprets.
"""

from typing import List, Dict, Optional

# Tunable thresholds for the rule-based tiers below. These are reasonable
# starting points, not measured truths — revisit once you have real usage
# data to see if "Good"/"High" etc. actually feel right to a student.
LEARNING_STATE_THRESHOLDS = {
    "min_resources_good": 3,
    "min_concepts_good": 3,
    "min_resources_moderate": 2,
    "min_concepts_moderate": 2,
    "confidence_high": 0.8,
    "confidence_medium": 0.5,
}

# A concept mentioned in this few resources (or fewer) counts as a gap.
RARE_CONCEPT_THRESHOLD = 1


def analyze_resource_usage(evidence: List[Dict]) -> Dict:
    """
    Which resources contributed to the answer, and how much each one
    contributed (measured by number of retrieved chunks per resource).
    """
    distribution = {
        e.get("resource_name", "Unknown"): len(e.get("retrieved_chunks", []))
        for e in evidence
    }

    if not distribution:
        return {"resources_used": 0, "primary_resource": None, "resource_distribution": {}}

    # primary = most chunks contributed; tie-break by retrieval_score
    score_by_name = {e.get("resource_name"): e.get("retrieval_score", 0) for e in evidence}
    primary_resource = max(
        distribution, key=lambda name: (distribution[name], score_by_name.get(name, 0))
    )

    return {
        "resources_used": len(distribution),  # always derived from distribution, can't drift out of sync
        "primary_resource": primary_resource,
        "resource_distribution": distribution,
    }


def analyze_concept_importance(knowledge_context: Dict) -> Dict:
    """Pass through the already-computed ranking from knowledge_structure.py —
    this function's role is just to select what's relevant for reflection."""
    return {
        "important_concepts": knowledge_context.get("main_concepts", []),
        "concept_frequency": knowledge_context.get("concept_frequency", {}),
    }


def detect_knowledge_gaps(
    knowledge_context: Dict, rare_threshold: int = RARE_CONCEPT_THRESHOLD
) -> Dict:
    """Concepts that appear in `rare_threshold` or fewer resources — thin
    coverage, likely worth flagging as a gap rather than a strength."""
    freq = knowledge_context.get("concept_frequency", {})
    gaps = [concept for concept, count in freq.items() if count <= rare_threshold]
    return {"knowledge_gaps": gaps}


def analyze_learning_state(evidence: List[Dict], knowledge_context: Dict) -> Dict:
    """
    Rule-based tiers from retrieval strength, resource count, and concept
    density. Thresholds live in LEARNING_STATE_THRESHOLDS at the top of
    this file — tune there, not inline here.
    """
    t = LEARNING_STATE_THRESHOLDS
    resources_used = len(evidence)
    concept_count = len(knowledge_context.get("main_concepts", []))
    retrieval_strength = evidence[0]["retrieval_score"] if evidence else 0.0

    if resources_used >= t["min_resources_good"] and concept_count >= t["min_concepts_good"]:
        coverage = "Good"
    elif resources_used >= t["min_resources_moderate"] and concept_count >= t["min_concepts_moderate"]:
        coverage = "Moderate"
    else:
        coverage = "Weak"

    if retrieval_strength >= t["confidence_high"]:
        confidence = "High"
    elif retrieval_strength >= t["confidence_medium"]:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "coverage": coverage,
        "confidence": confidence,
        "retrieval_strength": round(retrieval_strength, 4),
    }


def identify_patterns(evidence: List[Dict], knowledge_context: Dict) -> Dict:
    """
    Templated, deterministic observations from the computed stats — not
    LLM-generated. Gemma may rephrase these later; it doesn't invent them.
    """
    patterns = []

    usage = analyze_resource_usage(evidence)
    distribution = usage["resource_distribution"]
    if distribution:
        total = sum(distribution.values())
        top_resource, top_count = max(distribution.items(), key=lambda kv: kv[1])
        if total > 0 and top_count / total >= 0.5:
            patterns.append(f"Most evidence comes from {top_resource}.")

    freq = knowledge_context.get("concept_frequency", {})
    for concept, count in freq.items():
        if count > 1:
            patterns.append(f"{concept} appears across multiple resources.")

    return {"patterns": patterns}


def suggest_next_topics(
    knowledge_gaps: List[str], knowledge_context: Dict, max_topics: int = 5
) -> Dict:
    """
    knowledge_gaps + concepts related to the top concept (that aren't
    already well-covered) -> recommended next topics.
    """
    main_concepts = knowledge_context.get("main_concepts", [])
    related_concepts = knowledge_context.get("related_concepts", {})

    next_topics = list(dict.fromkeys(knowledge_gaps))  # dedupe, preserve order

    if main_concepts:
        top_concept = main_concepts[0]
        for related in related_concepts.get(top_concept, []):
            if related not in main_concepts and related not in next_topics:
                next_topics.append(related)

    return {"next_topics": next_topics[:max_topics]}


def build_reflection_context(
    query: str, evidence: List[Dict], knowledge_context: Dict
) -> Dict:
    """
    Final function — combines everything into ReflectionContext.
    `query` isn't used in the deterministic rules yet (nothing here is
    query-specific beyond what evidence/knowledge_context already encode
    from retrieval) — kept as a parameter for interaction logging and
    future query-specific reflection logic, not silently dropped.
    """
    resource_usage = analyze_resource_usage(evidence)
    concept_importance = analyze_concept_importance(knowledge_context)
    gaps = detect_knowledge_gaps(knowledge_context)
    learning_state = analyze_learning_state(evidence, knowledge_context)
    patterns = identify_patterns(evidence, knowledge_context)
    next_topics = suggest_next_topics(gaps["knowledge_gaps"], knowledge_context)

    return {
        "learning_state": learning_state,
        "resource_usage": {
            "resources_used": resource_usage["resources_used"],
            "primary_resource": resource_usage["primary_resource"],
        },
        "important_concepts": concept_importance["important_concepts"],
        "knowledge_gaps": gaps["knowledge_gaps"],
        "patterns": patterns["patterns"],
        "next_topics": next_topics["next_topics"],
    }


if __name__ == "__main__":
    import json

    sample_evidence = [
        {"resource_name": "Operating Systems.pdf", "retrieval_score": 0.91,
         "retrieved_chunks": [{}, {}, {}, {}]},
        {"resource_name": "Assignment3.pdf", "retrieval_score": 0.70,
         "retrieved_chunks": [{}, {}]},
    ]
    sample_knowledge_context = {
        "main_concepts": ["Deadlock", "Semaphore", "Mutex"],
        "concept_frequency": {"Deadlock": 4, "Semaphore": 3, "Mutex": 2, "Banker's Algorithm": 1},
        "related_concepts": {
            "Deadlock": ["Semaphore", "Mutex", "Banker's Algorithm", "Deadlock Prevention"],
        },
        "related_resources": ["Operating Systems.pdf", "Assignment3.pdf"],
    }

    context = build_reflection_context(
        "explain deadlocks", sample_evidence, sample_knowledge_context
    )
    print(json.dumps(context, indent=2))