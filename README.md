# ⚖️ Multi-Document Policy & Legal Q&A Agent (Hybrid RAG)

An enterprise-grade Retrieval-Augmented Generation (RAG) system built for querying multiple contracts, SOPs, and regulatory documents simultaneously with verifiable page-level citations and zero hallucinations.

## 🏗️ Architecture & Key Features
- **Metadata-Rich Multi-Format Ingestion (`ingest_engine.py`):** Extracts text page-by-page from PDFs (`pypdf`) and logical sections from Word documents (`python-docx`), attaching `source`, `page`, and `chunk_index` metadata to every overlapping chunk.
- **Hybrid Search with Reciprocal Rank Fusion (`hybrid_retriever.py`):** Combines **ChromaDB** dense semantic embeddings with **BM25** (`rank_bm25`) lexical keyword matching, fusing rankings via Reciprocal Rank Fusion (RRF) so exact clause numbers, currency figures, and dates are never missed.
- **Conversational Query Reformulation (`app.py`):** Rewrites follow-up questions using recent conversation context before retrieval.
- **Auditable Inline Citations & Evidence Viewer:** Enforces strict `[Source: Filename, Page/Section X]` inline citations and provides an expandable evidence inspector showing retrieved excerpts and RRF scores.
- **Multi-Model Fault Tolerance:** Automatic failover across Gemini model pools with strict HTTP timeouts.

## 🛠️ Tech Stack
- **Frontend & Deployment:** Streamlit, Streamlit Community Cloud
- **Vector & Lexical Retrieval:** ChromaDB, `rank_bm25` (Reciprocal Rank Fusion)
- **LLM Synthesis:** Google Gemini API (`google-genai` SDK)
- **Document Processing:** `pypdf`, `python-docx`

## 🚀 Local Setup
```bash
git clone [https://github.com/muhammadxel/policy-legal-rag-pro.git](https://github.com/muhammadxel/policy-legal-rag-pro.git)
cd policy-legal-rag-pro
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt