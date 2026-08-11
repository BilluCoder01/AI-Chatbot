import os
import time
import streamlit as st

# Import backend engine functions
from rag_engine import process_documents, stream_rag_pipeline

# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="CivilGPT | AI Engineering Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================
# ADVANCED CUSTOM CSS (Modern Dark Mode / Glassmorphism)
# ==============================================================
ADVANCED_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Theme Overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    /* Modern Hero Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 9999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.75rem;
        margin-bottom: 0;
        line-height: 1.6;
        max-width: 700px;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid #1e293b;
    }

    /* Sidebar Custom Cards */
    .sidebar-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Quick Action Buttons Hover Magic */
    div[data-testid="column"] .stButton button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid #334155;
        background: #0f172a;
        color: #e2e8f0;
        font-weight: 600;
        padding: 0.85rem 1rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    div[data-testid="column"] .stButton button:hover {
        border-color: #38bdf8;
        background: #1e293b;
        color: #38bdf8;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.15);
    }

    /* Expander / Source Citations Styling */
    .stExpander {
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        background: #0f172a !important;
        margin-top: 0.8rem !important;
    }

    /* Chat Avatar Polish */
    .stChatMessage {
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
</style>
"""
st.markdown(ADVANCED_CSS, unsafe_allow_html=True)

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
# SIDEBAR — DOCUMENT MANAGEMENT & STATUS
# ==============================================================
with st.sidebar:
    st.markdown("## 🏗️ CivilGPT")
    st.caption("AI Academic Assistant for Civil Engineers")
    st.divider()

    st.markdown("### 📚 Reference Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload course notes, textbooks, or IS Codes."
    )

    if uploaded_files:
        st.caption(f"📎 {len(uploaded_files)} file(s) selected")

    if st.button("⚙️ Process & Index Documents", type="primary", use_container_width=True, disabled=not uploaded_files):
        with st.spinner("Chunking text & generating Gemini embeddings..."):
            try:
                process_documents(uploaded_files)
                st.session_state.is_vector_db_ready = True
                st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
                st.toast("Vector database built successfully!", icon="✅")
            except Exception as e:
                st.error(f"Error processing PDFs: {e}")

    st.divider()

    # System Status Indicators
    st.markdown("### ⚡ System Status")
    if st.session_state.is_vector_db_ready:
        st.success("Knowledge Base: Active", icon="🟢")
        with st.expander("📄 Indexed Documents"):
            for name in st.session_state.uploaded_file_names:
                st.markdown(f"• `{name}`")
    else:
        st.warning("Knowledge Base: Not Loaded", icon="🟡")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================================
# HERO HEADER & METRICS METAPHOR
# ==============================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">⚡ Powered by Gemini 3.6 & ChromaDB</div>
        <h1 class="hero-title">CivilGPT Academic Copilot</h1>
        <p class="hero-subtitle">
            Ground your study sessions in validated engineering textbooks, design codes, and lecture notes. Ask complex structural, geotechnical, or fluid mechanics questions with exact page citations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Stats / Status Dashboard
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric(label="Model Engine", value="Gemini 3.6 Flash")
with m_col2:
    st.metric(label="Embedding Model", value="gemini-embedding-001")
with m_col3:
    st.metric(
        label="Database Status", 
        value="Ready" if st.session_state.is_vector_db_ready else "Idle",
        delta="Indexed" if st.session_state.is_vector_db_ready else "Awaiting PDF",
        delta_color="normal" if st.session_state.is_vector_db_ready else "off"
    )

st.divider()

# ==============================================================
# QUICK ACTIONS
# ==============================================================
st.markdown("#### 💡 Suggested Topics")

qa_col1, qa_col2, qa_col3 = st.columns(3)

quick_actions = [
    ("🧱 Explain Concrete Curing", "Explain the concrete curing process, its methods, and why it's critical for 28-day strength development."),
    ("📐 Summarize IS 456 Provisions", "Summarize the key design guidelines and provisions from IS 456 for reinforced concrete beams."),
    ("📊 Structural Deflection Methods", "Compare Moment Area Method and Conjugate Beam Method for calculating beam deflection."),
]

for col, (label, prompt) in zip([qa_col1, qa_col2, qa_col3], quick_actions):
    with col:
        if st.button(label, use_container_width=True, key=f"btn_{label}"):
            st.session_state.pending_prompt = prompt

st.divider()

# ==============================================================
# CHAT INTERFACE & RENDERING
# ==============================================================
if not st.session_state.messages:
    st.info("👋 Upload a PDF in the sidebar and ask a question, or click one of the topic cards above to start.", icon="💡")

# Render past chat history
for message in st.session_state.messages:
    avatar = "🧑‍🎓" if message["role"] == "user" else "🏗️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # Render source citations if attached to this message
        if message.get("sources"):
            with st.expander("📚 Referenced Sources & Context"):
                for src in message["sources"]:
                    st.markdown(f"• **{src['file']}** — *Page {src['page']}*")
                    st.caption(f"_{src['preview']}_")


def handle_user_message(prompt: str) -> None:
    """Handles prompt processing, memory pass, streaming answer, and citations."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.is_vector_db_ready:
        # Build chat history for memory context
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            role = "human" if msg["role"] == "user" else "ai"
            chat_history.append((role, msg["content"]))

        with st.chat_message("assistant", avatar="🏗️"):
            sources_container = []
            full_response = st.write_stream(stream_rag_pipeline(prompt, chat_history, sources_container))
            
            # Extract and format sources
            sources_data = []
            if sources_container:
                unique_sources = set()
                with st.expander("📚 Referenced Sources & Context"):
                    for doc in sources_container:
                        source_path = doc.metadata.get("source", "Unknown")
                        file_name = os.path.basename(source_path)
                        page = doc.metadata.get("page", 0) + 1
                        
                        source_id = f"{file_name}_pg{page}"
                        if source_id not in unique_sources:
                            unique_sources.add(source_id)
                            preview = doc.page_content[:150].replace("\n", " ") + "..."
                            
                            st.markdown(f"• **{file_name}** — *Page {page}*")
                            st.caption(f"_{preview}_")
                            
                            sources_data.append({
                                "file": file_name,
                                "page": page,
                                "preview": preview
                            })
    else:
        full_response = (
            "⚠️ Please upload and process a PDF document in the sidebar first so I can ground my answers in your study materials."
        )
        sources_data = []
        with st.chat_message("assistant", avatar="🏗️"):
            st.markdown(full_response)

    # Save complete assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": sources_data
    })


# Handle Quick Action click trigger
if st.session_state.pending_prompt:
    prompt_to_send = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    handle_user_message(prompt_to_send)
    st.rerun()

# Handle standard chat input
if user_prompt := st.chat_input("Ask CivilGPT about structural analysis, concrete design, IS codes..."):
    handle_user_message(user_prompt)
    st.rerun()