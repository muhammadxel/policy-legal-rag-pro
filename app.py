try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import time
import uuid
import streamlit as st
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

from ingest_engine import process_uploaded_files
from hybrid_retriever import HybridRetriever

# --- Page Configuration ---
st.set_page_config(
    page_title="Policy & Legal RAG Pro",
    page_icon="⚖️",
    layout="wide",
)

# --- Environment & Secrets ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it to your .env file or Streamlit Secrets.")
    st.stop()


# --- Initialize Session-Isolated Vector Store & Gemini Client ---
@st.cache_resource
def get_clients():
    chroma = chromadb.PersistentClient(path="./chroma_db")
    gemini = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=25000),
    )
    return chroma, gemini


chroma_client, gemini_client = get_clients()

if "session_collection_id" not in st.session_state:
    st.session_state.session_collection_id = f"session_{uuid.uuid4().hex[:12]}"

collection = chroma_client.get_or_create_collection(
    name=st.session_state.session_collection_id
)
retriever = HybridRetriever(collection)


def get_sample_legal_records() -> list[dict]:
    """Return pre-built multi-document legal & HR policy records for instant one-click demo."""
    return [
        {
            "id": "sample_msa_p1_c0",
            "text": (
                "MASTER SERVICES AGREEMENT (MSA-2026-09) — SECTION 1: PAYMENT TERMS & LATE PENALTIES. "
                "Client shall pay all undisputed invoices within thirty (30) calendar days of receipt (Net 30). "
                "Any overdue balance remaining unpaid after 30 days shall accrue late interest at 1.5% per month "
                "(18% per annum). If an invoice remains unpaid for more than forty-five (45) days, Vendor reserves "
                "the right to suspend all API and engineering deliverables upon 5 business days' written notice."
            ),
            "metadata": {
                "source": "Sample_Master_Services_Agreement.pdf",
                "page": 1,
                "chunk_index": 0,
                "file_type": "PDF",
            },
        },
        {
            "id": "sample_msa_p4_c0",
            "text": (
                "MASTER SERVICES AGREEMENT (MSA-2026-09) — SECTION 7: TERMINATION & IP OWNERSHIP. "
                "Either party may terminate this Agreement for convenience upon sixty (60) days' prior written notice. "
                "Upon termination, Client shall pay for all milestones completed up to the effective termination date. "
                "All custom AI models, vector pipelines, and source code deliverables vest exclusively in Client upon "
                "full settlement of final invoices, excluding Vendor's pre-existing proprietary utilities."
            ),
            "metadata": {
                "source": "Sample_Master_Services_Agreement.pdf",
                "page": 4,
                "chunk_index": 0,
                "file_type": "PDF",
            },
        },
        {
            "id": "sample_hr_p2_c0",
            "text": (
                "EMPLOYEE REMOTE WORK & DATA SECURITY POLICY (HR-POL-2026) — SECTION 3: CONFIDENTIALITY & AI TOOLS. "
                "Employees handling client contracts, personally identifiable information (PII), or proprietary datasets "
                "must never paste raw confidential records into unapproved public consumer AI chatbots. Only enterprise "
                "zero-data-retention API endpoints and encrypted local vector stores are authorized for document analysis."
            ),
            "metadata": {
                "source": "Sample_Employee_Security_Policy.docx",
                "page": 2,
                "chunk_index": 0,
                "file_type": "DOCX",
            },
        },
        {
            "id": "sample_hr_p5_c0",
            "text": (
                "EMPLOYEE REMOTE WORK & DATA SECURITY POLICY (HR-POL-2026) — SECTION 6: INCIDENT REPORTING & BREACHES. "
                "Any suspected loss of a company laptop, unauthorized API key exposure, or data leak must be reported "
                "to the Information Security Officer within two (2) hours of discovery. Failure to report a security "
                "incident within the 2-hour window may result in formal disciplinary action up to termination."
            ),
            "metadata": {
                "source": "Sample_Employee_Security_Policy.docx",
                "page": 5,
                "chunk_index": 0,
                "file_type": "DOCX",
            },
        },
    ]

retriever, gemini_client = init_system()


# --- Query Reformulation for Conversational Memory ---
def contextualize_query(user_query: str, chat_history: list[dict]) -> str:
    """Rewrite follow-up questions into standalone search queries using recent chat history."""
    if not chat_history:
        return user_query

    recent_turns = chat_history[-4:]  # Last 2 user-assistant pairs
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_turns
    )

    rewrite_prompt = f"""
Given the conversation history below and the user's latest question, rewrite the latest question into a standalone search query that resolves any pronouns or references (such as "that contract", "it", "the second policy", "what about penalties?").
Do NOT answer the question. Return ONLY the rewritten search query. If the question is already standalone, return it unchanged.

Conversation History:
{history_text}

Latest Question: {user_query}

Standalone Search Query:
"""
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=rewrite_prompt,
        )
        rewritten = (resp.text or "").strip()
        return rewritten if rewritten else user_query
    except Exception:
        return user_query


# --- Grounded Response Synthesis with Inline Citations ---
def generate_cited_response(
    user_query: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict],
) -> str:
    """Synthesize a strictly grounded legal/policy answer with mandatory inline source citations."""
    if not retrieved_chunks:
        return (
            "No indexed documents found matching your filter. "
            "Please upload and index PDF or DOCX files in the sidebar first."
        )

    formatted_sources = []
    for idx, item in enumerate(retrieved_chunks, start=1):
        meta = item["metadata"]
        source_name = meta.get("source", "Unknown")
        page_num = meta.get("page", "?")
        formatted_sources.append(
            f"[Excerpt {idx} | Source: {source_name}, Page/Section {page_num}]\n{item['text']}"
        )

    context_block = "\n\n---\n\n".join(formatted_sources)

    recent_turns = chat_history[-4:]
    history_block = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in recent_turns
    )

    system_prompt = f"""
You are a senior legal and policy compliance analyst. Answer the user's question accurately using ONLY the provided document excerpts below.

Strict Rules:
1. Every factual claim, clause summary, date, or obligation MUST include an inline citation formatted like this: ****.
2. If multiple sources support a point, cite all of them (e.g., **[Source: Contract_A.pdf, Page 2]**, ****).
3. If the provided excerpts do not contain enough information to answer the question, state clearly: "I cannot verify this from the currently indexed documents." Do NOT guess or use outside knowledge.
4. Maintain a clear, structured, professional tone using bullet points where appropriate.

Recent Conversation Context:
{history_block if history_block else "None"}

Retrieved Document Excerpts:
{context_block}

User Question:
{user_query}

Grounded Answer with Citations:
"""

    # Try multiple model pools automatically if one experiences a 503 traffic spike
   candidate_models = [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.8-flash-lite",
    ]
    last_error = ""

    for attempt in range(2):
        for model_name in candidate_models:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=system_prompt,
                )
                return response.text
            except Exception as e:
                last_error = str(e)
                if "404" in last_error:
                    continue
                time.sleep(1)

    return f"All Gemini model pools are currently busy (Last error: {last_error}). Please retry in a few seconds."

# --- Sidebar: Multi-Document Knowledge Base Manager ---
with st.sidebar:
    st.title("⚖️ Knowledge Base")
    st.caption("Hybrid Search (Vector + BM25) with Page-Level Citations")
    if st.button("⚡ Load Sample Legal Docs (Demo)", use_container_width=True):
        with st.spinner("Loading sample MSA & HR Security Policy..."):
            retriever.add_records(get_sample_legal_records())
            st.success("Loaded 2 sample documents! Try asking about payment penalties or breach reporting.")
            st.rerun()

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload Policy or Legal Documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process & Add to Index", type="primary", use_container_width=True):
            with st.spinner("Extracting metadata & building hybrid index..."):
                records, file_stats = process_uploaded_files(uploaded_files)
                if records:
                    retriever.add_records(records)
                    st.success(f"Indexed {len(records)} chunks across {len(file_stats)} file(s)!")
                else:
                    st.warning("No readable text found in the uploaded file(s).")

    st.markdown("---")
    st.subheader("📚 Indexed Documents")

    indexed_sources = retriever.get_indexed_sources()
    if indexed_sources:
        for doc_name, chunk_count in indexed_sources.items():
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{doc_name}**  \n`{chunk_count} chunks`")
            if col2.button("🗑️", key=f"del_{doc_name}", help=f"Remove {doc_name}"):
                retriever.delete_document(doc_name)
                st.rerun()

        st.markdown("")
        if st.button("Clear Entire Knowledge Base", use_container_width=True):
            retriever.clear_all()
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No documents indexed yet. Upload PDFs or DOCX files above.")

    st.markdown("---")
    st.subheader("🎛️ Retrieval Settings")
    source_options = ["All Documents"] + list(indexed_sources.keys())
    selected_source = st.selectbox("Filter Search Scope", options=source_options)
    top_k = st.slider("Top Retrieved Excerpts", min_value=2, max_value=10, value=5)


# --- Main Chat Interface ---
st.title("Multi-Document Policy & Legal Q&A Agent")
st.write(
    "Query across multiple contracts, SOPs, and regulatory files simultaneously with verifiable page citations."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render persistent conversation history + expandable source cards
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("🔍 View Retrieved Evidence & RRF Scores"):
                for idx, src in enumerate(msg["sources"], start=1):
                    meta = src["metadata"]
                    st.markdown(
                        f"**{idx}. {meta.get('source')} — Page/Section {meta.get('page')}** "
                        f"*(RRF Score: `{src.get('rrf_score')}`)*"
                    )
                    st.caption(src["text"])
                    if idx < len(msg["sources"]):
                        st.divider()

# Handle new user question
if user_prompt := st.chat_input("Ask a question across your policy or legal documents..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running hybrid retrieval (Semantic + BM25) & verifying citations..."):
            # Step 1: Contextualize follow-up queries
            standalone_query = contextualize_query(user_prompt, st.session_state.messages[:-1])

            # Step 2: Execute Hybrid RRF Search
            active_filter = None if selected_source == "All Documents" else selected_source
            retrieved = retriever.search(
                query=standalone_query,
                top_k=top_k,
                source_filter=active_filter,
            )

            # Step 3: Generate Cited Response
            answer = generate_cited_response(
                user_query=user_prompt,
                retrieved_chunks=retrieved,
                chat_history=st.session_state.messages[:-1],
            )
            st.markdown(answer)

            if retrieved:
                with st.expander("🔍 View Retrieved Evidence & RRF Scores"):
                    for idx, src in enumerate(retrieved, start=1):
                        meta = src["metadata"]
                        st.markdown(
                            f"**{idx}. {meta.get('source')} — Page/Section {meta.get('page')}** "
                            f"*(RRF Score: `{src.get('rrf_score')}`)*"
                        )
                        st.caption(src["text"])
                        if idx < len(retrieved):
                            st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved,
    })