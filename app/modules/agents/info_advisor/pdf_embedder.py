"""
pdf_embedder.py — Embeds the Python Developer Job Description PDF into ChromaDB.

Two public functions:
    build_vector_store() — run once (offline) to load, split, embed, and persist.
    get_retriever(k)     — load the existing Chroma store and return a retriever.

Files:
    Source PDF : data/Python Developer Job Description.pdf
    Chroma DB  : data/chroma_db/
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
_PDF_PATH     = _PROJECT_ROOT / "data" / "Python Developer Job Description.pdf"
_CHROMA_PATH  = _PROJECT_ROOT / "data" / "chroma_db"

_EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")


def build_vector_store() -> Chroma:
    """
    Load the job description PDF, split it into chunks, embed each chunk
    using OpenAI embeddings, and persist the result to a local Chroma DB.

    This is an offline/one-time operation. Run it once to populate the DB
    before calling get_retriever(). Re-running will overwrite existing data.

    Steps:
        1. PyPDFLoader reads the PDF page by page into LangChain Documents.
        2. RecursiveCharacterTextSplitter breaks them into 500-char chunks
           with 50-char overlap to preserve context across boundaries.
        3. OpenAIEmbeddings converts each chunk into a numeric vector using
           the text-embedding-3-small model.
        4. Chroma stores the vectors on disk at data/chroma_db/.

    Returns:
        Chroma: the populated vector store instance.
    """
    # Step 1 — load PDF pages as LangChain Document objects
    loader = PyPDFLoader(str(_PDF_PATH))
    docs   = loader.load()

    # Step 2 — split into overlapping chunks for better retrieval granularity
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)

    # Step 3 & 4 — embed chunks and persist to Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_EMBEDDINGS,
        persist_directory=str(_CHROMA_PATH),
    )
    print(f"Stored {len(chunks)} chunks in {_CHROMA_PATH}")
    return vectorstore


def get_retriever(k: int = 3):
    """
    Load the persisted Chroma vector store and return a LangChain retriever.

    The retriever converts a query string into a vector and returns the k
    most semantically similar chunks from the job description. Used by the
    Info Advisor to answer candidate questions about the position.

    Args:
        k (int): number of chunks to retrieve per query. Default is 3.
                 Increase for broader context, decrease for precision.

    Returns:
        VectorStoreRetriever: call .invoke("your question") to get results.

    Example:
        retriever = get_retriever(k=3)
        docs = retriever.invoke("What Python frameworks are required?")
    """
    # Auto-build the vector store on first use. Required for cold deployments
    # (e.g. Streamlit Community Cloud) where data/chroma_db/ is gitignored and
    # therefore absent from a freshly cloned repo.
    if not (_CHROMA_PATH / "chroma.sqlite3").exists():
        build_vector_store()

    vectorstore = Chroma(
        persist_directory=str(_CHROMA_PATH),
        embedding_function=_EMBEDDINGS,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
