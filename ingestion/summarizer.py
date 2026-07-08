"""
summarizer.py
Summarizes resource text using any LangChain-compatible chat model
(ChatOllama, or swap in Gemma later — same function either way).

Called after ingestion + chunking + embedding, in the pipeline:
extract -> chunk -> embed+store -> summarize (this file) -> db_writer.update_resource_summary
"""

from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from database.db_writer import update_resource_summary

SUMMARY_SYSTEM_PROMPT = (
    """You are building the semantic memory of a student's local AI assistant.

Your summary will NOT be shown to the student.

It will later be embedded and used for semantic retrieval.

Therefore:

- Mention every important academic concept.
- Mention important algorithms, formulas, definitions, frameworks.
- Mention important technologies if present.
- Mention important tools, APIs, libraries or software.
- Preserve technical terminology.
- Avoid motivational language.
- Avoid introductions.
- Avoid conclusions.
- Avoid vague phrases.

The goal is maximum retrieval quality rather than readability.

Return only the summary."""
)

COMBINE_SYSTEM_PROMPT = (
    "You are given several partial summaries of different sections of the "
    "same document. Combine them into a single concise 2-4 sentence summary "
    "covering the overall topics and concepts. Avoid repetition."
)


def _split_into_char_chunks(text: str, chunk_chars: int) -> List[str]:
    """Rough char-based split for map-reduce summarization — doesn't need
    to be semantically precise, just needs to fit the model's context."""
    return [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def _call_llm(llm, system_prompt: str, user_content: str) -> str:
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])
    return response.content.strip()


def summarize_resource_text(
    text: str,
    llm,
    max_direct_chars: int = 6000,
    chunk_chars: int = 3000,
) -> str:
    """
    Summarize resource text via a LangChain chat model.

    Args:
        text: full cleaned resource text (e.g. from extract_text_with_page_map).
        llm: any LangChain chat model instance (ChatOllama now, Gemma later —
             function doesn't care, just needs .invoke(messages)).
        max_direct_chars: if text is under this, summarize in one call.
        chunk_chars: chunk size used for map-reduce when text is longer than
                     max_direct_chars — keep this comfortably under whatever
                     context window your current test model actually has
                     (check num_ctx if using Ollama — defaults to 4K, not
                     the model's real max, unless you set it explicitly).

    Returns:
        summary: short text summary.
    """
    if not text or not text.strip():
        return ""

    if len(text) <= max_direct_chars:
        return _call_llm(llm, SUMMARY_SYSTEM_PROMPT, text)

    # map step: summarize each piece
    pieces = _split_into_char_chunks(text, chunk_chars)
    partial_summaries = [_call_llm(llm, SUMMARY_SYSTEM_PROMPT, p) for p in pieces]

    # reduce step: summarize the summaries
    combined_input = "\n\n".join(
        f"Section {i+1} summary: {s}" for i, s in enumerate(partial_summaries)
    )
    return _call_llm(llm, COMBINE_SYSTEM_PROMPT, combined_input)


def summarize_and_save(resource_id: str, text: str, llm, **kwargs) -> str:
    """
    Convenience wrapper: summarize, then write straight into resources.summary
    via db_writer.update_resource_summary. This is the function you call at
    the end of the ingestion pipeline.

    Returns:
        summary: the summary that was saved.
    """
    summary = summarize_resource_text(text, llm, **kwargs)
    update_resource_summary(resource_id, summary)
    return summary


if __name__ == "__main__":
    # Testing setup — swap ChatOllama's model string for Gemma later,
    # nothing else in this file needs to change.
    from langchain_ollama import ChatOllama
    from llm.llm_config import llm
    # llm = ChatOllama(model="llama3.2", num_ctx=8192)  # test model — swap to "gemma4:e2b" later

    sample_text = (
        "Deadlocks occur when a set of processes are blocked because each "
        "process is holding a resource and waiting for another resource "
        "acquired by some other process. The Banker's Algorithm is used for "
        "deadlock avoidance by simulating resource allocation and checking "
        "if the system remains in a safe state before granting a request."
    )

    summary = summarize_and_save("os_notes_pdf", sample_text, llm)
    print("Saved summary:", summary)