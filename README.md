# 🧠 Local Personal Intelligence

> **A privacy-first, agentic AI powered personal knowledge assistant for students.**

Local Personal Intelligence is an Agentic Retrieval-Augmented Generation (RAG) system designed to help students efficiently search across multiple study resources. Instead of manually opening several PDFs to locate a specific concept, students can simply ask a question and receive an accurate answer along with the exact source document from which the information was retrieved.

The entire system runs locally using open-source models, ensuring complete privacy while maintaining fast and efficient retrieval.

---

# 📖 Overview

Students often collect notes, lecture slides, textbooks, assignments, and documentation from multiple sources. As the number of study resources grows, finding a specific topic becomes increasingly time-consuming.

Local Personal Intelligence solves this problem by automatically indexing personal study material into a semantic knowledge base.

The workflow is simple:

* Upload one or more PDF documents.
* The system automatically extracts and indexes their contents.
* Ask questions in natural language.
* The system retrieves the most relevant information using semantic search.
* Answers are generated only from retrieved evidence.
* The original source document is returned alongside the answer for easy verification.

Rather than replacing study material, the system helps students quickly navigate their own knowledge base.

---

# ✨ Features

* 🔒 Completely local execution (no cloud APIs required)
* 🤖 Agent-based workflow using LangGraph
* 📄 Automatic PDF ingestion
* 🧩 Semantic chunking using similarity-threshold chunking
* 🧠 Two-stage semantic retrieval
* 📚 SQLite metadata management
* ⚡ ChromaDB vector storage
* 📝 LLM-generated resource summaries
* 📍 Source citation with document path
* 🖼️ Vision-based OCR fallback for failed PDF pages
* 🚫 Hallucination prevention through retrieval validation
* 🌐 Multilingual capability via local Qwen model
* 🔐 Privacy-first architecture

---

# ❓ Why This Project?

Traditional RAG systems usually retrieve similar chunks directly from an entire vector database.

As the number of documents increases, retrieval becomes slower and often less relevant.

This project introduces a **hierarchical retrieval pipeline**.

Instead of searching every chunk:

1. Relevant documents are first identified using summary embeddings.
2. Chunk-level retrieval is performed only within those shortlisted documents.
3. The retrieved evidence is transformed into structured knowledge before being sent to the reasoning model.

This significantly reduces unnecessary retrieval while improving answer relevance.

---

# 🏗 Architecture

The system follows an Agentic AI architecture where each agent performs one specialized responsibility.

```
                User Input
                     │
          ┌──────────┴──────────┐
          │                     │
      PDF Upload            User Query
          │                     │
          ▼                     ▼
     Supervisor Agent (Routing)
          │                     │
          │                     ▼
          │              Query Processing
          │                     │
          ▼                     ▼
     PDF Ingestion        Memory Agent
          │                     │
          ▼                     │
 Vision Recovery Agent          │
          │                     │
          ▼                     │
   Store Resource Metadata      │
          │                     │
          ▼                     ▼
   Knowledge Indexing      Evidence Builder
          │                     │
          ▼                     ▼
   Summarizer Agent     Knowledge Structure
          │                     │
          ▼                     ▼
 Summary Embedding      Reflection Context
          │                     │
          └──────────────┬──────┘
                         ▼
                  Reasoning Agent
                         │
                         ▼
               Answer + Source Citation
```

---

# ⚙ Workflow

## 1. Ingestion Pipeline

When a PDF is provided:

* Extract text using PyMuPDF.
* Automatically clean extracted text.
* Generate a unique Resource ID.
* Store PDF metadata in SQLite.
* Perform semantic chunking.
* Generate embeddings using **all-MiniLM-L6-v2**.
* Store chunk embeddings in ChromaDB.
* Summarize the document using the local LLM.
* Store summary embedding in a separate Chroma collection.

This prepares the document for efficient future retrieval.

---

## 2. Query Pipeline

When the user asks a question:

* Clean the user query.
* Generate semantic embedding.
* Search document summaries.
* Select the most relevant resources.
* Retrieve only relevant chunks.
* Build structured evidence.
* Generate knowledge relationships.
* Build reflection context.
* Pass structured memory to the reasoning model.
* Generate an evidence-grounded response.

---

# 🧠 Agent Responsibilities

## Supervisor Agent

Routes incoming requests into either the ingestion workflow or the query workflow.

---

## PDF Ingestion Node

Extracts text, computes file hashes, generates metadata, and prepares resources for indexing.

---

## Vision Recovery Agent

If PDF parsing fails or pages contain insufficient text, the page is rendered as an image and processed using the local vision-capable LLM.

This acts as an automatic OCR recovery mechanism.

---

## Knowledge Indexing Node

Responsible for:

* Semantic chunk generation
* Embedding creation
* ChromaDB storage

---

## Summarizer Agent

Creates concise semantic summaries describing the important topics contained in each document.

These summaries improve retrieval quality during the first retrieval stage.

---

## Memory Agent

The core retrieval orchestrator.

It performs:

* Resource retrieval
* Chunk retrieval
* Evidence construction
* Knowledge graph construction
* Reflection context generation

before sending structured memory to the reasoning agent.

---

## Reasoning Agent

Generates the final answer exclusively from retrieved evidence.

The response always includes the relevant source documents for verification.

---

# 🔍 Two-Stage Retrieval Strategy

Instead of searching every chunk in the vector database:

```
User Query
      │
      ▼
Summary Embedding Search
      │
      ▼
Relevant Resources
      │
      ▼
Chunk Search (Only Inside Those Resources)
      │
      ▼
Evidence Builder
```

This dramatically reduces search space and improves retrieval precision.

---

# 🛡 Failure Recovery Mechanisms

The system includes multiple recovery strategies to improve robustness.

### Duplicate Detection

Resources are identified using SHA-256 hashes, preventing duplicate ingestion.

---

### Vision OCR Recovery

If PDF parsing fails or extracted text is insufficient, the failed pages are rendered as images and processed using the local vision-capable LLM.

---

### Retrieval Validation

The system avoids hallucinations by checking:

* No chunks retrieved
* Similarity score below threshold (0.40)

In either case, the user receives a meaningful "No relevant information found" response instead of a fabricated answer.

---

### Transaction Safety

If ingestion fails midway, partially indexed data can be safely rolled back to maintain database consistency.

---

# 🗄 Knowledge Storage

The project combines relational storage with vector search.

### SQLite

Stores:

* Resource metadata
* Chunk metadata
* User interactions
* Concepts
* Relationships

---

### ChromaDB

Stores:

* Chunk embeddings
* Summary embeddings

This hybrid approach combines fast semantic search with structured metadata queries.

---

# 🛠 Tech Stack

### AI

* LangGraph
* LangChain
* Qwen 3.5 2B (Local LLM)
* Sentence Transformers

### Embedding Model

* all-MiniLM-L6-v2

### Vector Database

* ChromaDB

### Relational Database

* SQLite

### PDF Processing

* PyMuPDF
* pytesseract (fallback OCR)

### Programming Language

* Python

---

# 📂 Project Workflow

```
PDF
 │
 ▼
Extract
 │
 ▼
Chunk
 │
 ▼
Embed
 │
 ▼
Store
 │
 ▼
Summarize
 │
 ▼
Store Summary

==========================

Query
 │
 ▼
Embed
 │
 ▼
Retrieve Summary
 │
 ▼
Retrieve Chunks
 │
 ▼
Evidence Builder
 │
 ▼
Knowledge Structure
 │
 ▼
Reflection Context
 │
 ▼
Reasoning Agent
 │
 ▼
Answer
```

---

# 🔒 Privacy

One of the primary goals of this project is complete user privacy.

Unlike cloud-based assistants:

* Documents never leave the local machine.
* No external API calls are required.
* User knowledge is never used for model training.
* All inference is performed locally.

This makes the system suitable for personal notes, academic material, and sensitive documents.

---

# 🚀 Future Improvements

* Streamlit Web Interface
* Concept extraction pipeline
* Relationship graph visualization
* Multi-document comparative reasoning
* Incremental indexing
* Conversation memory
* Audio note ingestion
* Web article ingestion
* Personal study analytics

---

# 👨‍💻 Author

Developed as an **Agentic AI Capstone Project** demonstrating the integration of Retrieval-Augmented Generation, multi-agent orchestration, semantic search, local language models, and privacy-preserving AI systems.
