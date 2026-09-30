import os
import tempfile
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults # <-- NEW IMPORT
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ==========================================
# 1. CALCULATOR TOOL
# ==========================================
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression. 
    Use this for all math calculations. Example input: '45 * 10 / 2'
    """
    try:
        allowed_names = {}
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of {name} not allowed")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {e}"

# ==========================================
# 2. SEARCH TOOLS (DuckDuckGo & Tavily)
# ==========================================
duckduckgo_search = DuckDuckGoSearchRun(
    name="duckduckgo_search",
    description="Search the internet for current events, news, or up-to-date web information."
)

# Tavily is an AI-optimized search engine. It returns highly relevant snippets.
tavily_search = TavilySearchResults(
    max_results=3,
    name="tavily_search",
    description="Use this tool for comprehensive, accurate, and AI-optimized web searches."
)

# ==========================================
# 3. PDF RAG TOOL
# ==========================================
VECTOR_STORE = None

def process_pdf_from_frontend(uploaded_file_bytes: bytes):
    global VECTOR_STORE
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file_bytes)
        tmp_path = tmp.name
        
    try:
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        VECTOR_STORE = FAISS.from_documents(splits, embeddings)
    finally:
        os.remove(tmp_path)

@tool
def query_uploaded_pdf(query: str) -> str:
    """Use this tool to search the uploaded PDF document for information matching the query."""
    global VECTOR_STORE
    if VECTOR_STORE is None:
        return "No PDF document is currently loaded. Please ask the user to upload a PDF first."
    
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant information found in the uploaded document."
        
    return "\n\n".join([f"Excerpt:\n{d.page_content}" for d in docs])

# ==========================================
# EXPORT ALL TOOLS
# ==========================================
tools_list = [
    calculator, 
    duckduckgo_search, 
    tavily_search,  
    query_uploaded_pdf,
]