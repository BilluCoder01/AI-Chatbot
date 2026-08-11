import os
import shutil
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Load environment variables from the .env file
load_dotenv()

# 2. Grab the key explicitly so LangChain doesn't miss it
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configuration paths
DATA_DIR = "./data"
DB_DIR = "./db"

def _extract_text_from_chunk(content):
    """
    Safely extracts plain text strings whether content is a string
    or a list of structured block dictionaries.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return str(content) if content else ""

def process_documents(uploaded_files):
    """
    Takes Streamlit uploaded files, saves them temporarily, 
    extracts text, chunks it, and builds a Chroma vector database
    with rate-limit protection for Google's Free Tier API.
    """
    # Step 1: Clean up old data and database directories to start fresh
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR)

    all_docs = []
    
    # Step 2: Save uploaded files to disk and extract text page-by-page
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    # Step 3: Chunk the extracted text into manageable pieces
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_docs)

    # Step 4: Safety Check — Ensure text was actually extracted (handles scanned/empty PDFs)
    if not chunks:
        raise ValueError(
            "No text could be extracted from the uploaded PDF(s). "
            "Please ensure the file contains selectable text and is not a scanned image or password-protected."
        )

    # Step 5: Initialize Gemini Embeddings using the active gemini-embedding-001 model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    # Step 6: Initialize an empty Chroma vector database
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    
    # Step 7: Batch processing to prevent 429 RESOURCE_EXHAUSTED rate-limit errors
    batch_size = 5

    print(f"Total chunks to process: {len(chunks)}")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_db.add_documents(batch)
        print(f"Processed chunks {i} to {i + len(batch)}...")
        
        # Pause 10 seconds between batches to stay within Google's Free Tier rate limits
        time.sleep(10)
        
    return True

def get_retriever():
    """Returns the ChromaDB retriever so the LLM can search it."""
    # UPDATED: Using Google's new gemini-embedding-001 model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Return top 4 most relevant chunks for any query
    return vector_db.as_retriever(search_kwargs={"k": 4})

def query_rag_pipeline(user_query: str) -> str:
    """Queries the Chroma vector DB and generates an answer using Gemini."""
    
    # 1. Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3  # Keep it low for factual academic answers
    )

    # 2. Get the retriever we built in Step 2
    retriever = get_retriever()
    
    # 3. Create the System Prompt
    system_prompt = (
        "You are CivilGPT, a knowledgeable AI assistant for Civil Engineering students. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If the answer is not in the context, say so clearly, but still try to provide "
        "a general engineering answer if you know it. Keep your explanations clear, "
        "structured, and easy to study from.\n\n"
        "Context:\n{context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 4. Chain it all together
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 5. Ask the question and return the answer
    response = rag_chain.invoke({"input": user_query})
    return response["answer"]


def stream_general_chat(user_query: str, chat_history: list):
    """Streams a response directly from Gemini when no PDFs are uploaded."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.5
    )

    messages = [
        SystemMessage(content=(
            "You are CivilGPT, a knowledgeable AI assistant for Civil Engineering students. "
            "Provide clear, structured, and accurate engineering answers."
        ))
    ]
    
    for role, text in chat_history:
        if role == "human":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    
    messages.append(HumanMessage(content=user_query))

    for chunk in llm.stream(messages):
        # 🔑 Extract clean text from the chunk
        text = _extract_text_from_chunk(chunk.content)
        if text:
            yield text


def stream_rag_pipeline(user_query: str, chat_history: list, sources_container: list = None):
    """Queries ChromaDB with conversational memory and streams the answer."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are CivilGPT, a knowledgeable AI assistant for Civil Engineering students. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If the answer is not in the context, say so clearly, but still try to provide "
        "a general engineering answer if you know it. Keep your explanations clear, "
        "structured, and easy to study from.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    for chunk in rag_chain.stream({"input": user_query, "chat_history": chat_history}):
        if "context" in chunk and sources_container is not None:
            sources_container.extend(chunk["context"])
        
        if "answer" in chunk:
            # 🔑 Extract clean text from the chunk answer
            text = _extract_text_from_chunk(chunk["answer"])
            if text:
                yield text