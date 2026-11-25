import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from rag_engine import PDFRagEngine, HAS_SBERT
from sample_pdf import build_demo_pdf

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot with Page Citations",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Multi-PDF RAG Assistant with Exact Page Citations")
st.markdown(
    "Upload your own **PDF documents** (lecture notes, research papers, manuals), index them locally with "
    "**overlapping chunking & vector search**, and ask questions with **verified page-level citations**."
)

# Sidebar: PDF Upload & Chunking Controls
st.sidebar.header("Document Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=("pdf",),
    accept_multiple_files=True,
)
use_demo = st.sidebar.checkbox("Include Built-in AI Engineering Handbook PDF", value=True)
chunk_size = st.sidebar.slider("Chunk Size (chars)", min_value=250, max_value=1200, value=500, step=50)
top_k = st.sidebar.slider("Top-K Retrieved Chunks", min_value=1, max_value=6, value=3)

pdf_inputs = []
if use_demo:
    pdf_inputs.append(("AI_Engineering_Handbook.pdf", build_demo_pdf()))
if uploaded_files:
    for uf in uploaded_files:
        pdf_inputs.append((uf.name, uf.read()))

if not pdf_inputs:
    st.warning("Upload at least one PDF or keep the built-in handbook checked in the sidebar.")
    st.stop()

engine = PDFRagEngine(chunk_size=chunk_size, chunk_overlap=100)
total_chunks = engine.index_pdfs(pdf_inputs)

# KPI Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Indexed PDFs", len(pdf_inputs))
c2.metric("Total Chunks", total_chunks)
c3.metric("Retrieval Backend", "Dense SBERT + TF-IDF" if (HAS_SBERT and engine.embedder) else "Hybrid TF-IDF")
c4.metric("Top-K Contexts", top_k)

# Chat History Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Knowledge base indexed! Ask me anything about your uploaded PDFs (e.g. *'How do circuit breakers protect AI agents?'* or *'What is OWASP LLM01?'*).",
            "hits": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question grounded in your PDFs...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "hits": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer, hits = engine.generate_grounded_answer(prompt, top_k=top_k)
    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Show retrieval evidence & similarity chart for the latest query
latest_hits = next((m["hits"] for m in reversed(st.session_state.messages) if m["hits"]), [])
if latest_hits:
    st.divider()
    st.subheader("Retrieved Source Chunks & Similarity Scores")
    col_tbl, col_plot = st.columns((1.4, 1))
    df_hits = pd.DataFrame(latest_hits)
    with col_tbl:
        st.dataframe(df_hits, use_container_width=True)
    with col_plot:
        fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
        labels = [f"{r['Source'][:14]}.. (p.{r['Page']})" for _, r in df_hits.iterrows()]
        ax.barh(labels[::-1], df_hits["Relevance Score"][::-1], color="#1f77b4")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Cosine Similarity")
        ax.set_title("Chunk Retrieval Confidence")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
