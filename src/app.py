import os
import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

# Import backend engine functions
from rag_engine import process_documents, stream_rag_pipeline, stream_general_chat

# ==============================================================
# PAGE CONFIG & CSS
# ==============================================================
st.set_page_config(page_title="CivilGPT", page_icon="🏗️", layout="wide")

st.markdown("""

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

# Load animations (You can swap these URLs with any from LottieFiles.com)
lottie_robot = load_lottieurl("https://lottie.host/4a5b4c10-eb52-44df-b4a1-052be5ad38ec/N1qKz9lPzH.json")
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

# ==============================================================
# CUSTOM NAVIGATION MENU (Replaces the Sidebar)
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
    
    # Hero Section
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

    # Chat Rendering
    for message in st.session_state.messages:
        avatar = "🧑‍🎓" if message["role"] == "user" else "🏗️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Referenced Sources"):
                    for src in message["sources"]:
                        st.markdown(f"• **{src['file']}** — *Page {src['page']}*")

    def handle_user_message(prompt: str) -> None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Format chat history for LangChain
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
            
            # ROUTE 2: No Database (General AI Chat)
            else:
                st.caption("⚡ _Answering from general AI knowledge (No PDFs indexed)_")
                full_response = st.write_stream(stream_general_chat(prompt, chat_history))

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