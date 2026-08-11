import os
import shutil
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load environment variables from the .env file
load_dotenv()

# 2. Grab the key explicitly so LangChain doesn't miss it
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configuration paths
DATA_DIR = "./data"
DB_DIR = "./db"

def process_documents(uploaded_files):
    """
    Takes Streamlit uploaded files, saves them temporarily, 
    extracts text, chunks it, and builds a Chroma vector database.
    """
    # Clear out old data to avoid duplicates during prototyping
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR)

    all_docs = []
    
    # Save and Load PDFs
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    # Chunk the text
    # Civil Engineering docs have lots of tables and formulas. 
    # A chunk size of 1000 with 200 overlap keeps context intact.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_docs)

    # Generate Embeddings & Store in Chroma
    # UPDATED: Using Google's new gemini-embedding-001 model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
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


def stream_rag_pipeline(user_query: str):
    """
    Queries ChromaDB and yields answer tokens live as Gemini generates them.
    """
    # 1. Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    # 2. Get retriever
    retriever = get_retriever()

    # 3. Create System Prompt
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

    # 4. Chain setup
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. Stream answer chunks live
    for chunk in rag_chain.stream({"input": user_query}):
        if "answer" in chunk:
            yield chunk["answer"]