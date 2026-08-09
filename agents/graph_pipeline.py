import os
import io
import base64
import uuid
from typing import TypedDict, List, Dict, Any, Optional

import fitz
from PIL import Image
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from ingestion.pdf_ingestion import (
    extract_text_with_page_map, finalize_pages, clean_resume_text,
)
from ingestion.chunking import SemanticChunker
from ingestion.embed_store import embed_and_store_chunks, store_summary_embedding
from ingestion.summarizer import summarize_resource_text
from ingestion.encoder_config import encoder
from database.db_writer import save_resource, save_chunks, update_resource_summary, resource_exists
from database.chroma_client import collection, summary_collection
from utils.hashing import compute_file_hash
from llm.llm_config import llm

from retrieval.query_processing import process_query
from ingestion.resource_resolution import resolve_resources
from retrieval.hybrid_chunk_retrieval import hybrid_chunk_retrieval
from retrieval.reranker import load_reranker, rerank_chunks
from retrieval.relevance_gate import apply_relevance_gate
from retrieval.evidence_builder import build_evidence
from knowledge.knowledge_structure import build_knowledge_context
from reflection.reflection_context import build_reflection_context


class StudentMemoryState(TypedDict):
    input_type: str
    pdf_path: Optional[str]
    query: Optional[str]

    resource_id: Optional[str]
    resource_name: Optional[str]
    resource_type: Optional[str]
    file_hash: Optional[str]
    resource_exists: bool

    page_texts: Optional[List[str]]
    extracted_text: Optional[str]
    page_boundaries: Optional[List[Dict]]

    chunks: List[Dict]
    metadata: Dict[str, Any]

    cleaned_query: Optional[str]
    query_embedding: Optional[List[float]]

    retrieved_resources: List[Dict]
    retrieved_chunks: List[Dict]

    evidence: List[Dict]
    knowledge_context: Dict
    reflection_context: Dict
    memory_context: Dict

    response: Dict

    failed_pages: List[int]
    recovered_pages: List[int]
    ocr_recovery_used: bool

    retrieval_success: bool
    retrieval_failure_reason: Optional[str]
    relevance_threshold: float

    logs: List[str]


def create_initial_state(**overrides) -> StudentMemoryState:
    """Every list/dict field gets a real default here - the earlier version
    relied on the caller remembering to initialize each one, and missed
    fields (recovered_pages was the one that would've crashed first)."""
    state: StudentMemoryState = {
        "input_type": "",
        "pdf_path": None,
        "query": None,
        "resource_id": None,
        "resource_name": None,
        "resource_type": None,
        "file_hash": None,
        "resource_exists": False,
        "page_texts": None,
        "extracted_text": None,
        "page_boundaries": None,
        "chunks": [],
        "metadata": {},
        "cleaned_query": None,
        "query_embedding": None,
        "retrieved_resources": [],
        "retrieved_chunks": [],
        "evidence": [],
        "knowledge_context": {},
        "reflection_context": {},
        "memory_context": {},
        "response": {},
        "failed_pages": [],
        "recovered_pages": [],
        "ocr_recovery_used": False,
        "retrieval_success": False,
        "retrieval_failure_reason": None,
        "relevance_threshold": 0.5,
        "logs": [],
    }
    state.update(overrides)
    return state


VISION_SYSTEM_PROMPT = """
You are an OCR system.
Extract every visible word from the page.
Rules:
- Preserve equations.
- Preserve headings.
- Preserve bullet points.
- Do not summarize.
- Do not explain.
Return only extracted text.
"""

REASONING_SYSTEM_PROMPT = """
You are a Local Personal AI Assistant for students.

You are given:
1. User Question
2. Evidence retrieved from the student's personal knowledge base
3. Knowledge Context
4. Reflection Context

Your job is to answer ONLY using the retrieved evidence.

Instructions:
- Never hallucinate.
- If the retrieved evidence is insufficient, explicitly say so.
- Prefer retrieved chunks over summaries.
- Use summaries only for high-level understanding.
- Use Reflection Context when giving study suggestions.
- Keep answers educational and concise.
- Do NOT mention retrieval scores or internal system details.

Return only the answer.
"""


# ============================================================
# PDF ingestion nodes
# ============================================================

def pdf_ingestion_node(state: StudentMemoryState) -> StudentMemoryState:
    """Hash + dedup check happens FIRST, before the expensive extraction -
    the old version extracted+chunked+embedded a duplicate PDF fully
    before discovering it already existed."""
    pdf_path = state["pdf_path"]

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    file_hash = compute_file_hash(pdf_path)
    resource_id = f"res_{file_hash[:16]}"  # hash-based, not random - dedupes even if file gets renamed

    state["file_hash"] = file_hash
    state["resource_id"] = resource_id
    state["resource_name"] = os.path.basename(pdf_path)
    state["resource_type"] = "pdf"

    if resource_exists(file_hash):
        state["resource_exists"] = True
        state["logs"].append(f"Resource '{state['resource_name']}' already exists - skipping extraction.")
        return state

    state["resource_exists"] = False

    page_texts, failed_pages = extract_text_with_page_map(pdf_path)
    state["page_texts"] = page_texts
    state["failed_pages"] = failed_pages

    state["logs"].append(
        f"PDF '{state['resource_name']}' extracted ({len(failed_pages)} pages need recovery)."
    )
    return state


def vision_recovery_agent(state: StudentMemoryState) -> StudentMemoryState:
    """Fills failed pages IN PLACE in page_texts - page_boundaries hasn't
    been computed yet at this point (finalize_pages runs after this node),
    so there's no stale-offset risk like the append-to-end version had."""
    pdf_path = state["pdf_path"]
    doc = fitz.open(pdf_path)
    page_texts = state["page_texts"]

    for page_no in state["failed_pages"]:
        page = doc.load_page(page_no)
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # proper multimodal message - raw PIL Image isn't valid input to .invoke()
        response = llm.invoke([
            SystemMessage(content=VISION_SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": "Extract all text from this page image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]),
        ])

        page_texts[page_no] = response.content.strip()
        state["recovered_pages"].append(page_no)

    doc.close()
    state["page_texts"] = page_texts
    state["ocr_recovery_used"] = True
    state["logs"].append(f"Vision recovery filled {len(state['recovered_pages'])} pages.")
    return state


def finalize_extraction_node(state: StudentMemoryState) -> StudentMemoryState:
    """The ONLY place page_boundaries gets computed - always after any
    vision recovery, never before, never twice."""
    full_text, page_boundaries = finalize_pages(state["page_texts"])
    state["extracted_text"] = full_text
    state["page_boundaries"] = page_boundaries
    state["logs"].append("Extraction finalized, page boundaries computed.")
    return state


def store_resource_node(state: StudentMemoryState) -> StudentMemoryState:
    save_resource(
        resource_id=state["resource_id"],
        name=state["resource_name"],
        resource_type=state["resource_type"],
        file_hash=state["file_hash"],
        path=state["pdf_path"],
    )
    state["logs"].append("Resource metadata stored.")
    return state


def knowledge_indexing_node(state: StudentMemoryState) -> StudentMemoryState:
    resource_id = state["resource_id"]
    text = state["extracted_text"]
    page_boundaries = state["page_boundaries"]

    chunker = SemanticChunker(encoder=encoder)  # shared encoder, not a freshly loaded one
    chunks = chunker.chunk(text=text, resource_id=resource_id)

    embed_and_store_chunks(chunks=chunks, resource_id=resource_id, collection=collection, encoder=encoder)
    save_chunks(chunks=chunks, resource_id=resource_id, page_boundaries=page_boundaries)

    state["chunks"] = chunks
    state["logs"].append(f"{len(chunks)} chunks indexed.")
    return state


def summarizer_agent_node(state: StudentMemoryState) -> StudentMemoryState:
    summary = summarize_resource_text(text=state["extracted_text"], llm=llm)
    state["metadata"] = {"summary": summary}
    state["logs"].append("Resource summarized successfully.")
    return state


def summary_saver_node(state: StudentMemoryState) -> StudentMemoryState:
    resource_id = state["resource_id"]
    summary = state["metadata"]["summary"]
    update_resource_summary(resource_id, summary)
    store_summary_embedding(summary_text=summary, resource_id=resource_id, collection=summary_collection, encoder=encoder)
    state["logs"].append("Summary saved successfully.")
    return state


def already_exists_node(state: StudentMemoryState) -> StudentMemoryState:
    state["logs"].append("Skipped - resource already ingested.")
    return state


# ============================================================
# Query workflow nodes
# ============================================================

def query_processing_node(state: StudentMemoryState) -> StudentMemoryState:
    cleaned_query, embedding = process_query(query=state["query"], encoder=encoder)
    state["cleaned_query"] = cleaned_query
    state["query_embedding"] = embedding
    state["logs"].append("Query processed successfully.")
    return state


_reranker = None
def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = load_reranker()
    return _reranker


def memory_agent_node(state: StudentMemoryState) -> StudentMemoryState:
    """V2 retrieval: resource resolution (RRF of semantic+lexical) ->
    hybrid chunk retrieval (RRF of semantic+lexical) -> rerank -> gate.
    Replaces the old semantic-only search_resources_by_summary /
    search_chunks_by_resource_ids pipeline entirely."""
    cleaned_query = state["cleaned_query"]
    query_embedding = state["query_embedding"]

    resource_ids = resolve_resources(cleaned_query, query_embedding, summary_collection, top_k=5)
    state["retrieved_resources"] = resource_ids

    if not resource_ids:
        state["retrieval_success"] = False
        state["retrieval_failure_reason"] = "NO_RESOURCES"
        return state

    candidates = hybrid_chunk_retrieval(cleaned_query, query_embedding, resource_ids, collection)
    if not candidates:
        state["retrieval_success"] = False
        state["retrieval_failure_reason"] = "NO_CHUNKS"
        return state

    reranked = rerank_chunks(cleaned_query, candidates, _get_reranker())
    gated_chunks = apply_relevance_gate(reranked, threshold=state.get("relevance_threshold", 0.5))

    state["retrieved_chunks"] = gated_chunks

    if not gated_chunks:
        state["retrieval_success"] = False
        state["retrieval_failure_reason"] = "BELOW_RELEVANCE_THRESHOLD"
        return state

    state["retrieval_success"] = True

    evidence = build_evidence(gated_chunks)
    state["evidence"] = evidence

    knowledge_context = build_knowledge_context(evidence)
    state["knowledge_context"] = knowledge_context

    reflection_context = build_reflection_context(query=cleaned_query, evidence=evidence, knowledge_context=knowledge_context)
    state["reflection_context"] = reflection_context

    state["memory_context"] = {
        "evidence": evidence,
        "knowledge_context": knowledge_context,
        "reflection_context": reflection_context,
    }
    state["logs"].append("Memory context built successfully (V2 hybrid+rerank pipeline).")
    return state


def reasoning_agent_node(state: StudentMemoryState) -> StudentMemoryState:
    memory_context = state["memory_context"]

    prompt = f"""
User Question
-------------
{state["cleaned_query"]}

Evidence
--------
{memory_context["evidence"]}

Knowledge Context
-----------------
{memory_context["knowledge_context"]}

Reflection Context
------------------
{memory_context["reflection_context"]}
"""

    response = llm.invoke([
        SystemMessage(content=REASONING_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    sources = []
    seen = set()
    for e in memory_context["evidence"]:
        if e["resource_id"] in seen:
            continue
        seen.add(e["resource_id"])
        sources.append({"resource_name": e["resource_name"], "resource_path": e["resource_path"]})

    state["response"] = {"answer": response.content.strip(), "sources": sources}
    state["logs"].append("Reasoning Agent completed successfully.")
    return state


def no_results_node(state: StudentMemoryState) -> StudentMemoryState:
    state["response"] = {
        "answer": (
            "I couldn't find sufficiently relevant information in your personal knowledge base.\n\n"
            "Try:\n- Rephrasing the question\n- Uploading related notes\n- Asking a broader question"
        ),
        "sources": [],
    }
    state["logs"].append(f"No results: {state.get('retrieval_failure_reason')}")
    return state  # was missing entirely before - state update was silently dropped


def supervisor_node(state: StudentMemoryState) -> StudentMemoryState:
    if state.get("pdf_path"):
        state["input_type"] = "pdf"
    elif state.get("query"):
        state["input_type"] = "query"
    state["logs"].append("Supervisor executed.")
    return state


# ============================================================
# Routers (pure functions - just read state, return the next node name)
# ============================================================

def route_supervisor(state: StudentMemoryState) -> str:
    return "pdf_ingestion" if state["input_type"] == "pdf" else "query_processing"


def pdf_ingestion_router(state: StudentMemoryState) -> str:
    if state.get("resource_exists"):
        return "already_exists"
    if state.get("failed_pages"):
        return "vision_recovery"
    return "finalize_extraction"


def retrieval_router(state: StudentMemoryState) -> str:
    return "reasoning" if state["retrieval_success"] else "no_results"


# ============================================================
# Graph construction
# ============================================================

builder = StateGraph(StudentMemoryState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("pdf_ingestion", pdf_ingestion_node)
builder.add_node("vision_recovery", vision_recovery_agent)
builder.add_node("finalize_extraction", finalize_extraction_node)
builder.add_node("store_resource", store_resource_node)
builder.add_node("knowledge_indexing", knowledge_indexing_node)
builder.add_node("summarizer", summarizer_agent_node)
builder.add_node("summary_saver", summary_saver_node)
builder.add_node("already_exists", already_exists_node)

builder.add_node("query_processing", query_processing_node)
builder.add_node("memory_agent", memory_agent_node)
builder.add_node("reasoning_agent", reasoning_agent_node)
builder.add_node("no_results", no_results_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_supervisor, {
    "pdf_ingestion": "pdf_ingestion",
    "query_processing": "query_processing",
})

# PDF ingestion path - dedup check routes around all the expensive work
builder.add_conditional_edges("pdf_ingestion", pdf_ingestion_router, {
    "already_exists": "already_exists",
    "vision_recovery": "vision_recovery",
    "finalize_extraction": "finalize_extraction",
})
builder.add_edge("already_exists", END)
builder.add_edge("vision_recovery", "finalize_extraction")  # recovery always feeds into finalize, never skips it
builder.add_edge("finalize_extraction", "store_resource")
builder.add_edge("store_resource", "knowledge_indexing")
builder.add_edge("knowledge_indexing", "summarizer")
builder.add_edge("summarizer", "summary_saver")
builder.add_edge("summary_saver", END)

# Query path
builder.add_edge("query_processing", "memory_agent")
builder.add_conditional_edges("memory_agent", retrieval_router, {
    "reasoning": "reasoning_agent",
    "no_results": "no_results",
})
builder.add_edge("reasoning_agent", END)
builder.add_edge("no_results", END)

graph = builder.compile()


if __name__ == "__main__":
    state = create_initial_state(query="How are the Hackathon Tasks in Devcation hackathon")
    result = graph.invoke(state)
    print(result["response"]["answer"])
    for src in result["response"]["sources"]:
        print(src["resource_name"], src["resource_path"])