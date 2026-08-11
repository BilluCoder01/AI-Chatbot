import os
import json
import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import re
from io import BytesIO
from docx import Document
import markdown
import asyncio
import nest_asyncio
from pdf_generator import generate_pdf_bytes

# Import backend engine functions (including the fallback generator)
from rag_engine import process_documents, stream_rag_pipeline, stream_general_chat

# ==============================================================
# PAGE CONFIG & CSS
# ==============================================================
st.set_page_config(page_title="CivilGPT", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
    .stExpander { border: 1px solid #334155 !important; border-radius: 12px !important; }
    div[data-testid="column"] .stButton button {
        border-radius: 12px;
        border: 1px solid #334155;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="column"] .stButton button:hover {
        border-color: #38bdf8;
        color: #38bdf8;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================
# HELPER FUNCTION: LOAD LOTTIE ANIMATIONS
# ==============================================================
@st.cache_data
def load_lottieurl(url: str):
    """Fetches a Lottie animation from a URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
lottie_docs = load_lottieurl("https://lottie.host/804d9c73-f93d-4228-b09e-716d2cf7bc51/J3c0cWqVzH.json")

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

def clean_ai_formatting(text: str) -> str:
    """Removes markdown symbols like **, *, ###, and ` so text looks human-written."""
    # Remove bold and italic asterisks/underscores
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    # Remove header hashes
    text = re.sub(r'#+\s*', '', text)
    # Remove inline code backticks
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text.strip()

# ==============================================================
# UTILITIES SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("### ⚙️ Utilities")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    if st.session_state.messages:
        st.divider()
        st.markdown("### 💾 Export Notes")
        
        # One-Click PDF Generation
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Rendering tables and math formulas..."):
                try:
                    # Call the Playwright backend
                    pdf_data = asyncio.run(generate_pdf_bytes(st.session_state.messages))
                    
                    st.download_button(
                        label="⬇️ Click to Download PDF",
                        data=pdf_data,
                        file_name="CivilGPT_Study_Notes.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("PDF Ready!")
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")

# ==============================================================
# CUSTOM NAVIGATION MENU
# ==============================================================
selected_tab = option_menu(
    menu_title=None,
    options=["💬 CivilGPT Chat", "📚 Knowledge Base"],
    icons=["robot", "server"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "#38bdf8", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#1e293b"},
        "nav-link-selected": {"background-color": "#0f172a"},
    }
)

# ==============================================================
# TAB 1: KNOWLEDGE BASE (Document Uploading)
# ==============================================================
if selected_tab == "📚 Knowledge Base":
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        if lottie_docs:
            st_lottie(lottie_docs, height=300, key="docs_animation")
        else:
            st.write("📚")
            
    with col2:
        st.title("Build Your Knowledge Base")
        st.write("Upload your structural engineering notes, textbooks, and IS Codes here. CivilGPT will read them and use them to answer your questions.")
        
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        
        if st.button("⚙️ Process & Index Documents", type="primary", use_container_width=True, disabled=not uploaded_files):
            with st.spinner("Chunking text & generating Gemini embeddings (this takes a moment)..."):
                try:
                    process_documents(uploaded_files)
                    st.session_state.is_vector_db_ready = True
                    st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
                    st.success("Vector database built successfully! Head over to the Chat tab.")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")
        
        if st.session_state.is_vector_db_ready:
            st.divider()
            st.success(f"🟢 Database Active: {len(st.session_state.uploaded_file_names)} document(s) indexed.")

# ==============================================================
# TAB 2: AI CHAT INTERFACE
# ==============================================================
if selected_tab == "💬 CivilGPT Chat":
    
    # Hero Section (Only shows if chat is empty)
    if not st.session_state.messages:
        st.markdown("## 🏗️ CivilGPT Academic Copilot")
        st.write("Ground your study sessions in validated engineering textbooks. Ask complex questions below with exact page citations.")
        
        # Quick Actions
        qa_col1, qa_col2, qa_col3 = st.columns(3)
        quick_actions = [
            ("🧱 Concrete Curing", "Explain the concrete curing process and its importance."),
            ("📐 IS 456 Provisions", "Summarize key IS 456 guidelines for reinforced concrete."),
            ("📊 Beam Deflection", "Compare methods for calculating beam deflection."),
        ]
        for col, (label, prompt) in zip([qa_col1, qa_col2, qa_col3], quick_actions):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state.pending_prompt = prompt

        st.divider()

    # Chat Rendering Loop
    for message in st.session_state.messages:
        avatar = "🧑‍🎓" if message["role"] == "user" else "🏗️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Referenced Sources"):
                    for src in message["sources"]:
                        st.markdown(f"• **{src['file']}** — *Page {src['page']}*")

    def handle_user_message(prompt: str) -> None:
        """Handles RAG pipeline (if PDFs exist) or General Chat fallback."""
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Format history for LangChain
        chat_history = [( "human" if msg["role"] == "user" else "ai", msg["content"] ) for msg in st.session_state.messages[:-1]]
        sources_data = []
        
        with st.chat_message("assistant", avatar="🏗️"):
            # ROUTE 1: Database is ready (RAG Pipeline)
            if st.session_state.is_vector_db_ready:
                sources_container = []
                full_response = st.write_stream(stream_rag_pipeline(prompt, chat_history, sources_container))
                
                if sources_container:
                    unique_sources = set()
                    with st.expander("📚 Referenced Sources"):
                        for doc in sources_container:
                            file_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                            page = doc.metadata.get("page", 0) + 1
                            source_id = f"{file_name}_pg{page}"
                            
                            if source_id not in unique_sources:
                                unique_sources.add(source_id)
                                st.markdown(f"• **{file_name}** — *Page {page}*")
                                sources_data.append({"file": file_name, "page": page})
            
            # ROUTE 2: No Database (General AI Chat Fallback)
            else:
                st.caption("⚡ _Answering from general AI knowledge (No PDFs indexed)_")
                full_response = st.write_stream(stream_general_chat(prompt, chat_history))

        # Save to session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "sources": sources_data
        })

    # Handle Triggers
    if st.session_state.pending_prompt:
        prompt_to_send = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        handle_user_message(prompt_to_send)
        st.rerun()

    if user_prompt := st.chat_input("Ask CivilGPT about structural analysis, concrete design..."):
        handle_user_message(user_prompt)
        st.rerun()