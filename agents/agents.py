import fitz
from langgraph.graph import  add_messages
from langgraph.graph import StateGraph,START,END

from typing import TypedDict, List, Dict, Any, Optional

from IPython.display import Image
from scipy import io
from ingestion.pdf_ingestion import extract_text_from_pdf,clean_resume_text,extract_text_with_page_map
from database.db_writer import save_chunks
import uuid
from utils.hashing import compute_file_hash


from typing import TypedDict, Optional, List, Dict, Any


class StudentMemoryState(TypedDict):

    # --------------------------
    # Input
    # --------------------------

    input_type: str                  # pdf | query

    pdf_path: Optional[str]

    query: Optional[str]


    # --------------------------
    # Resource Metadata
    # --------------------------

    resource_id: Optional[str]

    resource_name: Optional[str]

    resource_type: Optional[str]

    file_hash: Optional[str]


    # --------------------------
    # Extracted Document
    # --------------------------

    extracted_text: Optional[str]

    page_boundaries: Optional[List[Dict]]


    # --------------------------
    # Chunking
    # --------------------------

    chunks: List[Dict]


    # --------------------------
    # Metadata (Gemma Output)
    # --------------------------

    metadata: Dict[str, Any]
    """
    {
        summary,
        concepts,
        keywords
    }
    """


    # --------------------------
    # Query Processing
    # --------------------------

    cleaned_query: Optional[str]

    query_embedding: Optional[List[float]]


    # --------------------------
    # Retrieval
    # --------------------------

    retrieved_resources: List[Dict]

    retrieved_chunks: List[Dict]


    # --------------------------
    # Evidence
    # --------------------------

    evidence: List[Dict]


    # --------------------------
    # Knowledge
    # --------------------------

    knowledge_context: Dict


    # --------------------------
    # Reflection
    # --------------------------

    reflection_context: Dict


    # --------------------------
    # Memory Agent Output
    # --------------------------

    memory_context: Dict


    # --------------------------
    # Final Output
    # --------------------------

   
    response:Dict

    # ---------- Failure Handling ----------
    transaction_active: bool

    ingestion_success: bool

    failure_reason: str | None

    failed_node: str | None

    # ---------- Rollback ----------
    rollback_required: bool

    # ---------- OCR Recovery ----------
    ocr_recovery_used: bool

    failed_pages: List[int]

    recovered_pages: List[int]

    # ---------- Retrieval ----------
    retrieval_success: bool
    retrieval_failure_reason: str | None
    similarity_threshold:float
    # --------------------------
    # Debug
    # --------------------------

    logs: List[str]



def pdf_ingestion_node(state: StudentMemoryState) -> StudentMemoryState:
    """
    PDF Ingestion Node

    Responsibilities:
    -----------------
    - Validate PDF path
    - Generate resource_id
    - Extract text
    - Clean extracted text
    - Extract page boundaries
    - Compute file hash
    - Extract resource metadata
    - Populate graph state
    """

    pdf_path = state["pdf_path"]

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Generate unique resource ID
    resource_id = f"res_{uuid.uuid4().hex[:12]}"

    # Extract text + page mapping
    extracted_text, page_boundaries, failed_pages = extract_text_with_page_map(pdf_path)

    # Clean text
    cleaned_text = clean_resume_text(extracted_text)

    # Compute file hash
    file_hash = compute_file_hash(pdf_path)

    # Update graph state
    state["resource_id"] = resource_id
    state["resource_name"] = os.path.basename(pdf_path)
    state["resource_type"] = "pdf"
    state["file_hash"] = file_hash
    state["failed_pages"] = failed_pages
    state["extracted_text"] = cleaned_text
    state["page_boundaries"] = page_boundaries

    state["logs"].append(
        f"PDF '{state['resource_name']}' ingested successfully."
    )

    return state

#knowldege indexing node
#chunking on the extracted text and storing the chunks in the state


from ingestion.chunking import SemanticChunker
from ingestion.embed_store import embed_and_store_chunks

from database.chroma_client import collection
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
encoder = SentenceTransformer("all-MiniLM-L6-v2",token=os.getenv("HUGGING_FACE_TOKEN"))

chunker = SemanticChunker(
    embedding_model="all-MiniLM-L6-v2")



def knowledge_indexing_node(state: StudentMemoryState):

    resource_id = state["resource_id"]

    text = state["extracted_text"]

    page_boundaries = state["page_boundaries"]


    # -------------------
    # Chunking
    # -------------------

    chunker = SemanticChunker()

    chunks = chunker.chunk(
        text=text,
        resource_id=resource_id
    )


    # -------------------
    # Store Chunk Embeddings
    # -------------------

    embed_and_store_chunks(

        chunks=chunks,

        resource_id=resource_id,

        collection=collection,

        encoder=encoder

    )


    # -------------------
    # Store Chunks SQLite
    # -------------------

    save_chunks(

        chunks=chunks,

        resource_id=resource_id,

        page_boundaries=page_boundaries

    )


    state["chunks"] = chunks

    state["logs"].append(
        f"{len(chunks)} chunks indexed."
    )

    return state

#ingestion routing node
def ingestion_recovery_router(state: StudentMemoryState):

    # Entire extraction failed
    if state.get("failure_reason"):
        return "vision_recovery"

    # Some pages failed
    if state.get("failed_pages"):
        return "vision_recovery"

    return "store_resource"
#storing pdf data in db
"""
store_resource_node.py

LangGraph Node

Responsibilities:
-----------------
- Store initial PDF metadata in SQLite.
- Resource row is created before chunk indexing.
- Summary is intentionally left NULL.
"""

import os



from database.db_writer import save_resource,resource_exists


def store_resource_node(state: StudentMemoryState) -> StudentMemoryState:
    """
    Stores the resource metadata into SQLite.

    Expected State Inputs:
    ----------------------
    resource_id
    pdf_path
    resource_name
    resource_type
    file_hash

    Updates:
    --------
    logs
    """
def store_resource_node(state: StudentMemoryState):

    if resource_exists(state["file_hash"]):

        state["logs"].append(
            "Resource already exists. Skipping ingestion."
        )

        state["resource_exists"] = True

        return state

    save_resource(
        resource_id=state["resource_id"],
        name=state["resource_name"],
        resource_type=state["resource_type"],
        file_hash=state["file_hash"],
        path=state["pdf_path"],
    )

    state["resource_exists"] = False

    state["logs"].append(
        "Resource metadata stored."
    )

    return state

#Vision agent 



SYSTEM_PROMPT = """
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


def vision_recovery_agent(state: StudentMemoryState):

    pdf = fitz.open(state["pdf_path"])

    recovered_text = []

    for page_no in state["failed_pages"]:

        page = pdf.load_page(page_no)

        pix = page.get_pixmap(dpi=300)

        img = Image.open(
            io.BytesIO(pix.tobytes("png"))
        )

        response = llm.invoke(
            img
        )

        recovered_text.append(response.content)

        state["recovered_pages"].append(page_no)

    pdf.close()

    state["ocr_recovery_used"] = True

    state["extracted_text"] += "\n".join(recovered_text)

    return state


#summarizing llm agent
"""
summarizer_agent.py

LangGraph Node

Responsibilities
----------------
- Use Gemma to summarize the resource.
- Produce retrieval-oriented summary.
- Extract important concepts.
- Extract important keywords.

Does NOT store anything.
"""


from ingestion.summarizer import summarize_resource_text
from llm.llm_config import llm


def summarizer_agent_node(
    state: StudentMemoryState
) -> StudentMemoryState:

    text = state["extracted_text"]

    summary = summarize_resource_text(
        text=text,
        llm=llm
    )

    state["metadata"] = {

        "summary": summary

    }

    state["logs"].append(
        "Resource summarized successfully."
    )

    return state


#summary storage node
"""
summary_saver_node.py

Responsibilities

- Update SQLite summary

- Store summary embedding
"""



from database.db_writer import update_resource_summary

from ingestion.embed_store import store_summary_embedding

from database.chroma_client import summary_collection


def summary_saver_node(
    state: StudentMemoryState
) -> StudentMemoryState:

    resource_id = state["resource_id"]

    summary = state["metadata"]["summary"]


    # -----------------------
    # SQLite
    # -----------------------

    update_resource_summary(

        resource_id,

        summary

    )


    # -----------------------
    # Chroma
    # -----------------------

    store_summary_embedding(

        summary_text=summary,

        resource_id=resource_id,

        collection=summary_collection

    )


    state["logs"].append(
        "Summary saved successfully."
    )

    return state

#=============================Query Workflow Nodes=============================
#query processing node


from retrieval.query_processing import process_query

from ingestion.encoder_config import encoder


def query_processing_node(
    state: StudentMemoryState,
) -> StudentMemoryState:
    """
    Process the user's query for semantic retrieval.

    Expected Inputs
    ----------------
    state["query"]

    Updates
    ----------------
    state["cleaned_query"]
    state["query_embedding"]
    """

    cleaned_query, embedding = process_query(
        query=state["query"],
        encoder=encoder,
    )

    state["cleaned_query"] = cleaned_query
    state["query_embedding"] = embedding

    state["logs"].append(
        "Query processed successfully."
    )

    return state

#memory agent node --> memory context


from retrieval.resource_search import search_resources_by_summary
from retrieval.chunk_search import search_chunks_by_resource_ids

from retrieval.evidence_builder import build_evidence

from knowledge.knowledge_structure import build_knowledge_context

from reflection.reflection_context import build_reflection_context

from database.chroma_client import (
    summary_collection,
    collection,
)


def memory_agent_node(
    state: StudentMemoryState,
) -> StudentMemoryState:
    """
    Build MemoryContext from indexed student knowledge.
    """

    # ---------------------------------------
    # Stage 1 : Resource Retrieval
    # ---------------------------------------

    resources = search_resources_by_summary(
        query_embedding=state["query_embedding"],
        collection=summary_collection,
        top_k=5,
    )

    state["retrieved_resources"] = resources

    resource_ids = [
        r["resource_id"]
        for r in resources
    ]

    # ---------------------------------------
    # Stage 2 : Chunk Retrieval
    # ---------------------------------------

    chunks = search_chunks_by_resource_ids(
        query_embedding=state["query_embedding"],
        resource_ids=resource_ids,
        collection=collection,
        top_k=10,
    )

    state["retrieved_chunks"] = chunks
    if len(chunks) == 0:
     state["retrieval_success"] = False
     state["retrieval_failure_reason"] = "NO_CHUNKS"
     return state

    best_score = chunks[0]["score"]

    threshold = state.get("similarity_threshold", 0.40)

    if best_score < threshold:
        state["retrieval_success"] = False
        state["retrieval_failure_reason"] = "LOW_SIMILARITY"
        return state

    state["retrieval_success"] = True

    # ---------------------------------------
    # Evidence Builder
    # ---------------------------------------

    evidence = build_evidence(chunks)

    state["evidence"] = evidence

    # ---------------------------------------
    # Knowledge Structure
    # ---------------------------------------

    knowledge_context = build_knowledge_context(
        evidence
    )

    state["knowledge_context"] = knowledge_context

    # ---------------------------------------
    # Reflection Context
    # ---------------------------------------

    reflection_context = build_reflection_context(
        query=state["cleaned_query"],
        evidence=evidence,
        knowledge_context=knowledge_context,
    )

    state["reflection_context"] = reflection_context

    # ---------------------------------------
    # Memory Context
    # ---------------------------------------

    state["memory_context"] = {

        "evidence": evidence,

        "knowledge_context": knowledge_context,

        "reflection_context": reflection_context,

    }

    state["logs"].append(
        "Memory context built successfully."
    )

    return state

#Reasoning agent node


from langchain_core.messages import SystemMessage, HumanMessage

from llm.llm_config import llm



SYSTEM_PROMPT = """
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


def reasoning_agent_node(
    state: StudentMemoryState,
) -> StudentMemoryState:

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

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

        # -----------------------------
    # Collect Source PDFs
    # -----------------------------

    sources = []

    seen = set()

    for evidence in memory_context["evidence"]:

        if evidence["resource_id"] in seen:
            continue

        seen.add(evidence["resource_id"])

        sources.append(
            {
                "resource_name": evidence["resource_name"],
                "resource_path": evidence["resource_path"],
            }
        )

    # -----------------------------
    # Final Response
    # -----------------------------

    state["response"] = {
        "answer": response.content.strip(),
        "sources": sources,
    }

    state["logs"].append(
        "Reasoning Agent completed successfully."
    )
    
    return state

#supervisor router agent node
"""
supervisor_agent.py

Entry point of the application.

Responsibilities
----------------
- Determine the user's input type.
- Route execution to the correct workflow.

It does NOT perform any business logic.
It does NOT call the LLM.
It does NOT generate resource IDs.

Those responsibilities belong to downstream nodes.
"""

import os

 


def supervisor_node(state: StudentMemoryState):

    if state.get("pdf_path"):
        state["input_type"] = "pdf"

    elif state.get("query"):
        state["input_type"] = "query"

    state["logs"].append("Supervisor executed.")

    return state

def route_supervisor(state: StudentMemoryState):

    if state["input_type"] == "pdf":
        return "pdf_ingestion"

    return "query_processing"

def retrieval_router(state):

    if state["retrieval_success"]:
        return "reasoning"

    return "no_results"

def no_results_node(state: StudentMemoryState):
    state["response"] = {

    "answer":
    "I couldn't find sufficiently relevant information in your personal knowledge base.\n\n"
    "Try:\n"
    "- Rephrasing the question\n"
    "- Uploading related notes\n"
    "- Asking a broader question",

    "sources":[]
}
    

#graph construction
builder = StateGraph(StudentMemoryState)

#nodes


builder.add_node(
    "pdf_ingestion",
    pdf_ingestion_node,
)

builder.add_node(
    "store_resource",
    store_resource_node,
)

builder.add_node(
    "knowledge_indexing",
    knowledge_indexing_node,
)

builder.add_node(
    "summarizer",
    summarizer_agent_node,
)

builder.add_node(
    "summary_saver",
    summary_saver_node,
)

builder.add_node(
    "query_processing",
    query_processing_node,
)

builder.add_node(
    "memory_agent",
    memory_agent_node,
)

builder.add_node(
    "reasoning_agent",
    reasoning_agent_node,
)




builder.add_node(
    "vision_recovery",
    vision_recovery_agent,
)

builder.add_node(
    "no_results",
    no_results_node,
)


#edges

#PDF Edges======================================================


builder.add_edge(
    "store_resource",
    "knowledge_indexing",
)

builder.add_edge(
    "knowledge_indexing",
    "summarizer",
)

builder.add_edge(
    "summarizer",
    "summary_saver",
)

builder.add_edge(
    "summary_saver",
    END,
)



builder.add_conditional_edges(
    "pdf_ingestion",
    ingestion_recovery_router,
    {
        "store_resource": "store_resource",
        "vision_recovery": "vision_recovery",
    },
)

builder.add_edge(
    "vision_recovery",
    "store_resource",
)



def route_resource(state):

    if state["resource_exists"]:
        return END

    return "knowledge_indexing"


#Query Edges======================================================
builder.add_edge(
    "query_processing",
    "memory_agent",
)

builder.add_edge(
    "memory_agent",
    "reasoning_agent",
)

builder.add_edge(
    "reasoning_agent",
    END,
)

builder.add_edge(
    START,
    "supervisor",
)

builder.add_node(
    "supervisor",
    supervisor_node
)

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "pdf_ingestion": "pdf_ingestion",
        "query_processing": "query_processing",
    }
)

builder.add_conditional_edges(

    "memory_agent",

    retrieval_router,

    {

        "reasoning":"reasoning_agent",

        "no_results":"no_results",

    }

)

builder.add_edge(
    "no_results",
    END,
)


#compile the graph
graph = builder.compile()



if __name__ == "__main__":
    #  state = {
        
    #     "query": "How are the Hackathon Tasks in Devcation hackathon",

    #     "logs": [],
    #     "pdf_path": "",  # optional defaults
    #     # optional defaults
    #     "response": {},
    # }

    #  result = graph.invoke(state)
    # #  print("\n========== LOGS ==========\n")
    # #  for log in result["logs"]:
    # #     print(log)
    #  print("\n========== ANSWER ==========\n")
    #  print(result["response"]["answer"])

    #  print("\n========== SOURCES ==========\n")
    #  for src in result["response"]["sources"]:
    #     print(src["resource_name"])
    #     print(src["resource_path"])
    #     print()
    
 

     with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

     print("Graph saved as graph.png")