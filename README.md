# 🧠 Local Personal Intelligence

> **A Privacy-First Multi-Agent Personal Knowledge Assistant for Students**

Local Personal Intelligence is a **local-first, Agentic Retrieval-Augmented Generation (RAG)** system that transforms scattered study materials into a searchable personal knowledge base.

Instead of manually searching through multiple PDFs, students can simply ask questions in natural language. The system retrieves the most relevant information from their personal documents, generates an evidence-grounded answer using a local language model, and cites the original source document.

Unlike cloud-based AI assistants, **all processing happens locally**, ensuring complete privacy while providing intelligent semantic search and retrieval.

---

# 📖 Overview

Students often maintain hundreds of pages of lecture notes, assignments, documentation, textbooks, and reference PDFs. As this collection grows, finding a specific concept becomes increasingly time-consuming.

Local Personal Intelligence addresses this problem by automatically ingesting, indexing, and organizing PDF documents into a semantic knowledge base.

The workflow is simple:

1. Upload one or more PDF documents.
2. The system extracts and indexes their contents.
3. Ask questions in natural language.
4. Relevant documents are retrieved using semantic search.
5. The system generates an answer grounded in the retrieved evidence.
6. The original source documents are returned alongside the answer.

Rather than replacing study material, the project helps students efficiently navigate their own knowledge repository.

---

# ✨ Features

## 📚 Knowledge Base

* Automatic PDF ingestion
* Semantic document chunking
* Intelligent document summarization
* Persistent knowledge storage

## 🔍 Intelligent Retrieval

* Two-stage semantic retrieval
* Summary-based resource filtering
* Chunk-level semantic search
* Structured evidence generation
* Reflection-aware reasoning
* Source citation with document path

## 🤖 Multi-Agent Architecture

* Supervisor Agent
* Memory Agent
* Reasoning Agent
* Vision Recovery Agent
* Summarizer Agent

## 🔒 Privacy First

* Fully local execution
* Local LLM (Qwen)
* No cloud API dependency
* No user data leaves the machine

---

# ❓ Problem Statement

Students frequently spend more time locating information than actually learning it.

Searching across multiple lecture notes, books, documentation, and assignments is inefficient because:

* Information is scattered.
* Traditional keyword search lacks semantic understanding.
* Similar concepts are expressed differently across resources.
* Students often forget which document contains the required information.

This project converts personal study material into an intelligent semantic memory that understands concepts rather than exact keywords.

---

# 🏗 System Architecture

The system follows a modular multi-agent architecture.

```text
                    User Input
                         │
              ┌──────────┴──────────┐
              │                     │
          PDF Path              User Query
              │                     │
              ▼                     ▼
          Supervisor Agent (Routing)
              │                     │
              │                     ▼
              │             Query Processing
              │                     │
              ▼                     ▼
        PDF Ingestion          Memory Agent
              │                     │
              ▼                     ▼
     Ingestion Recovery       Reasoning Agent
              │                     │
      ┌───────┴────────┐            │
      │                │            │
      ▼                ▼            │
 Vision Recovery   Store Resource   │
              │                │     │
              └────────┬───────┘     │
                       ▼             │
               Knowledge Indexing    │
                       │             │
                       ▼             │
                Summarizer Agent     │
                       │             │
                       ▼             │
                Summary Saver        │
                       │             │
                       └─────────────┘
                             │
                             ▼
                    Final Response
```

<img width="574" height="853" alt="final_graph" src="https://github.com/user-attachments/assets/0de7cf88-de2a-444f-83c1-e7694aa43de5" />



---

# 📥 Ingestion Workflow

When a PDF is provided, the Supervisor Agent routes the request to the ingestion workflow.

The ingestion pipeline performs:

1. PDF validation
2. Resource ID generation
3. Text extraction using PyMuPDF
4. PDF metadata extraction
5. SQLite resource storage
6. Semantic chunk generation
7. Embedding generation using **all-MiniLM-L6-v2**
8. ChromaDB storage
9. LLM-based document summarization
10. Summary embedding generation
11. Summary storage for retrieval

After completion, the document becomes part of the student's personal knowledge base.

---

# 🔎 Query Workflow

When the user asks a question, the query pipeline performs:

1. Query cleaning
2. Query embedding generation
3. Summary retrieval
4. Resource filtering
5. Chunk retrieval
6. Evidence construction
7. Knowledge context generation
8. Reflection context generation
9. Retrieval validation
10. Reasoning using the local LLM
11. Response generation with source citation

---

# 🧠 Memory Agent

The Memory Agent is the core retrieval orchestrator.

Instead of directly passing retrieved chunks to the language model, it builds a structured **Memory Context**.

The Memory Agent performs the following deterministic stages:

### Stage 1 — Resource Retrieval

Searches the summary embedding collection to identify the most relevant resources.

---

### Stage 2 — Chunk Retrieval

Searches only the shortlisted resources instead of the entire vector database.

---

### Stage 3 — Evidence Builder

Transforms retrieved chunks into structured Evidence Objects containing:

* Resource name
* Resource summary
* Retrieved chunks
* Retrieval score

---

### Stage 4 — Knowledge Context

Builds structured knowledge information from the retrieved evidence.

---

### Stage 5 — Reflection Context

Generates a deterministic reflection context describing:

* Learning state
* Resource usage
* Important concepts
* Knowledge gaps
* Learning patterns
* Suggested next topics

---

The complete Memory Context is then passed to the Reasoning Agent.

---

# 🎯 Two-Stage Retrieval Strategy

Rather than searching every chunk inside the vector database, retrieval occurs in two stages.

```text
User Query
      │
      ▼
Summary Embedding Search
      │
      ▼
Relevant Resources
      │
      ▼
Chunk Embedding Search
      │
      ▼
Evidence Builder
      │
      ▼
Reasoning
```

### Why?

Searching every chunk becomes increasingly expensive as more documents are added.

By first retrieving relevant resources using summary embeddings, chunk retrieval is limited to only those documents.

This reduces the search space while improving retrieval quality.

---

# 🤖 Reasoning Agent

The Reasoning Agent is responsible for generating the final response.

It receives:

* User query
* Evidence Object
* Knowledge Context
* Reflection Context

Using this structured memory, the local language model produces an answer grounded entirely in retrieved evidence.

The response also includes the source document names and file paths, allowing students to verify and revisit the original material.

---

# 🛡 Failure Handling

The current implementation includes two failure recovery mechanisms.

## 1. Vision OCR Recovery

Some PDF pages may fail during extraction or contain insufficient text.

Instead of failing the ingestion process, the affected pages are rendered as images and processed by the local vision-capable language model.

The recovered text is appended before indexing, ensuring that valuable content is not lost.

---

## 2. Retrieval Validation

After chunk retrieval, the Memory Agent validates the results.

If:

* no chunks are retrieved, or
* the highest similarity score is below the configured threshold,

the Reasoning Agent is skipped.

Instead, the user receives a meaningful response indicating that no sufficiently relevant information was found.

This prevents the language model from generating unsupported or hallucinated answers.

---

# 🗄 Data Storage

The project combines relational storage with semantic vector search.

## SQLite

Stores:

* Resource metadata
* Chunk metadata
* User interactions
* Document summaries

## ChromaDB

Stores:

* Chunk embeddings
* Summary embeddings

This hybrid design combines efficient metadata storage with fast semantic retrieval.

---

# 🛠 Tech Stack

| Category               | Technology                    |
| ---------------------- | ----------------------------- |
| Programming Language   | Python                        |
| Workflow Orchestration | LangGraph                     |
| LLM Framework          | LangChain                     |
| Local Language Model   | Qwen (via Ollama)             |
| Embedding Model        | all-MiniLM-L6-v2              |
| Vector Database        | ChromaDB                      |
| Metadata Database      | SQLite                        |
| PDF Parsing            | PyMuPDF                       |
| OCR Recovery           | Vision-capable Qwen           |
| Semantic Chunking      | Similarity Threshold Chunking |

---

# 📂 Project Structure

```text
Local-Personal-Intelligence
│
├── agents/
├── database/
├── ingestion/
├── retrieval/
├── llm/
├── chroma/
├── app.py
└── README.md
```

---

# 💡 Design Decisions

## Local LLM

Using a local language model ensures complete privacy, eliminates API costs, and allows offline operation.

---

## Semantic Chunking

Instead of fixed-length chunking, semantic chunking preserves coherent concepts, leading to more meaningful retrieval.

---

## Summary Embeddings

Generating document summaries enables efficient first-stage retrieval before searching individual chunks.

---

## Two-Stage Retrieval

Searching summaries before chunk embeddings significantly reduces unnecessary vector search and improves scalability.

---

## Deterministic Memory Pipeline

Evidence construction, knowledge organization, and reflection are computed deterministically before invoking the LLM.

This minimizes hallucinations by ensuring the model reasons over structured evidence instead of raw retrieval results.

---

# 🚀 Future Improvements

* Frontend inegration
* Concept extraction pipeline
* Relationship graph visualization
* Better sources retrival separation 
* Support for additional document formats
* Conversation memory
* Personal study analytics
* voice support and multilanguage support
* remove the hugging face api interaction use internet rather download them.

---

# 🔒 Privacy

Privacy is a primary objective of this project.

* All inference runs locally.
* Documents never leave the user's device.
* No external APIs are required.
* Personal study material is never used for model training.

---
# 🚀 Setup Guide

## 1. Clone the Repository

```bash
git clone <repository-url>
cd Local-Personal-Intelligence
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv nvenv
nvenv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv nvenv
source nvenv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Download and install Ollama from:

https://ollama.com/

Pull the required model:

```bash
ollama pull qwen3:4b
```

> Replace the model name if your project is configured to use a different local model.

---

## 5. Install Tesseract OCR

Required only for OCR fallback during PDF ingestion.

Download and install Tesseract OCR:

https://github.com/tesseract-ocr/tesseract

After installation, update the Tesseract executable path in the project if required.

---

## 6. Initialize the Database

Run the database initialization script:

```bash
python -m database.schema
```

This creates the required SQLite tables.

---

## 7. Run the Application

```bash
python input_output.py
```

---

# 🧪 Testing the Project

### Ingest a PDF

Select:

```
1. Ingest PDF
```

Then provide the absolute path to a PDF:

```
C:\Users\YourName\Documents\Notes.pdf
```

The system will:

* Extract text
* Build semantic chunks
* Generate embeddings
* Store metadata
* Create document summaries

---

### Ask Questions

Select:

```
2. Ask Question
```

Example queries:

* What is deadlock?
* Explain Banker's Algorithm.
* What are the deliverables of the hackathon?
* How does row rank relate to pivots?

The system returns:

* Evidence-grounded answer
* Source document names
* Source file paths

---

## Project Requirements

* Python 3.10 or newer
* Ollama installed and running
* Qwen model downloaded
* Tesseract OCR installed (for OCR fallback)
* Internet connection required only for downloading models during the initial setup and hugging face api interaction with transformers

Once configured, the project runs entirely offline using local models.



# 👨‍💻 Author

Developed as an **Agentic AI Capstone Project** demonstrating the practical application of multi-agent systems, Retrieval-Augmented Generation (RAG), semantic search, local language models, and privacy-preserving AI for academic knowledge management.
