"""
CivilGPT — RAG-based AI Assistant for Civil Engineering Students
------------------------------------------------------------------
UI LAYER ONLY.

This file defines the complete Streamlit front-end for CivilGPT.
All retrieval-augmented generation logic (document parsing, chunking,
embedding, vector storage, and LLM calls) is intentionally left as
placeholders so the backend team can wire it in independently.
"""

import time
import streamlit as st

# TODO: Import rag_engine here
from rag_engine import process_documents , stream_rag_pipeline
#     get_vector_db_status,
#     query_rag_pipeline,
# )


# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="CivilGPT | AI Study Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================
# CUSTOM CSS
# ==============================================================
CUSTOM_CSS = """
<style>
    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* ---------- Hero Header ---------- */
    .civilgpt-hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 18px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 32, 39, 0.35);
        position: relative;
        overflow: hidden;
    }
    .civilgpt-hero::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,183,77,0.18) 0%, rgba(255,183,77,0) 70%);
        border-radius: 50%;
    }
    .civilgpt-hero h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .civilgpt-hero p {
        color: #cfe3ea;
        font-size: 1.02rem;
        margin-top: 0.55rem;
        margin-bottom: 0;
        max-width: 640px;
        line-height: 1.5;
    }
    .civilgpt-badge {
        display: inline-block;
        background: rgba(255, 183, 77, 0.16);
        color: #ffb74d;
        border: 1px solid rgba(255, 183, 77, 0.35);
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-bottom: 0.9rem;
    }

    /* ---------- Section labels ---------- */
    .section-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #6b7280;
        margin-bottom: 0.4rem;
        margin-top: 0.2rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }

    /* ---------- Status pill ---------- */
    .status-pill {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 0.9rem;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-top: 0.6rem;
    }
    .status-ready {
        background: rgba(52, 211, 153, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .status-pending {
        background: rgba(251, 191, 36, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    .dot {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        display: inline-block;
        background: currentColor;
    }

    /* ---------- Quick action cards ---------- */
    div[data-testid="column"] .stButton button {
        width: 100%;
        height: 100%;
        min-height: 78px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        color: #1f2937;
        font-weight: 600;
        text-align: left;
        padding: 0.9rem 1rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        transition: all 0.15s ease-in-out;
    }
    div[data-testid="column"] .stButton button:hover {
        border-color: #2c5364;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.10);
        transform: translateY(-2px);
        color: #0f2027;
    }

    /* ---------- Chat bubbles spacing ---------- */
    .stChatMessage {
        margin-bottom: 0.4rem;
    }

    /* ---------- Footer note ---------- */
    .civilgpt-footer-note {
        text-align: center;
        color: #9ca3af;
        font-size: 0.78rem;
        margin-top: 1.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================
# SESSION STATE INITIALIZATION
# ==============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_vector_db_ready" not in st.session_state:
    st.session_state.is_vector_db_ready = False

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ==============================================================
# SIDEBAR — DOCUMENT MANAGEMENT
# ==============================================================
with st.sidebar:
    st.markdown("### 🏗️ CivilGPT")
    st.caption("Your Civil Engineering knowledge base, on demand.")
    st.divider()

    st.markdown('<div class="section-label">Upload Reference Material</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDFs (IS Codes, textbooks, notes)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or more PDF documents to build your knowledge base.",
    )

    if uploaded_files:
        st.caption(f"📎 {len(uploaded_files)} file(s) selected")

    process_col = st.container()
    with process_col:
        process_clicked = st.button(
            "⚙️ Process Documents",
            use_container_width=True,
            type="primary",
            disabled=not uploaded_files,
        )

    if process_clicked and uploaded_files:
        with st.spinner("Reading documents and building vector index..."):
            # Call our new function!
            process_documents(uploaded_files)
            
        st.session_state.is_vector_db_ready = True
        st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
        st.toast("Vector database ready!", icon="✅")

    st.divider()
    st.markdown('<div class="section-label">Knowledge Base Status</div>', unsafe_allow_html=True)

    if st.session_state.is_vector_db_ready:
        st.markdown(
            """
            <div class="status-pill status-ready">
                <span class="dot"></span> Vector DB Ready
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.uploaded_file_names:
            with st.expander("Indexed documents"):
                for name in st.session_state.uploaded_file_names:
                    st.markdown(f"• {name}")
    else:
        st.markdown(
            """
            <div class="status-pill status-pending">
                <span class="dot"></span> No Documents Indexed Yet
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<div class="civilgpt-footer-note">Built for structural, geotechnical &amp; '
        'construction engineering study.</div>',
        unsafe_allow_html=True,
    )


# ==============================================================
# HERO HEADER
# ==============================================================
st.markdown(
    """
    <div class="civilgpt-hero">
        <div class="civilgpt-badge">🎓 RAG-Powered Study Assistant</div>
        <h1>CivilGPT 🏗️</h1>
        <p>
            Ask questions grounded in your own IS Codes, textbooks, and lecture notes.
            Upload your reference PDFs on the left, then chat with an assistant that
            actually cites your materials — built for Civil Engineering students.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================
# QUICK ACTIONS
# ==============================================================
st.markdown('<div class="section-label">Quick Actions</div>', unsafe_allow_html=True)

qa_col1, qa_col2, qa_col3 = st.columns(3, gap="medium")

quick_actions = [
    ("🧱 Explain Concrete Curing", "Explain the concrete curing process, its methods, and why it's critical for strength development."),
    ("📘 Summarize IS 456 Guidelines", "Summarize the key design guidelines and provisions from IS 456 for reinforced concrete structures."),
    ("📐 Review Structural Analysis", "Review the fundamentals of structural analysis, including common methods for determinate and indeterminate structures."),
]

for col, (label, prompt) in zip([qa_col1, qa_col2, qa_col3], quick_actions):
    with col:
        if st.button(label, use_container_width=True, key=f"qa_{label}"):
            st.session_state.pending_prompt = prompt

st.divider()


# ==============================================================
# CHAT INTERFACE
# ==============================================================
st.markdown('<div class="section-label">Chat</div>', unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info(
            "👋 Ask a question about your uploaded materials, or try one of the "
            "quick actions above to get started.",
            icon="💡",
        )

    for message in st.session_state.messages:
        avatar = "🧑‍🎓" if message["role"] == "user" else "🏗️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


def handle_user_message(prompt: str) -> None:
    """Append user message and stream the assistant's reply live."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.is_vector_db_ready:
        # Display the assistant avatar and stream the response
        with st.chat_message("assistant", avatar="🏗️"):
            # st.write_stream handles token-by-token rendering and returns the complete text
            full_response = st.write_stream(stream_rag_pipeline(prompt))
    else:
        full_response = (
            "⚠️ I don't have any indexed documents yet. "
            "Please upload and process a PDF in the sidebar first so I can ground "
            "my answers in your course materials."
        )
        with st.chat_message("assistant", avatar="🏗️"):
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Handle a quick-action click from the previous rerun
if st.session_state.pending_prompt:
    prompt_to_send = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    handle_user_message(prompt_to_send)
    st.rerun()

# Handle normal chat input
user_prompt = st.chat_input("Ask CivilGPT about concrete design, IS codes, structural analysis...")
if user_prompt:
    handle_user_message(user_prompt)
    st.rerun()