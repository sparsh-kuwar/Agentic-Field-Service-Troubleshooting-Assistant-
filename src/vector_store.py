from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


PDF_PATH = "data/manuals/X200_Industrial_Water_Pump_Manual.pdf"
VECTORSTORE_PATH = "vectorstore"


def create_vector_store():
    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating FAISS vector store...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local(VECTORSTORE_PATH)

    print(f"Vector store saved to '{VECTORSTORE_PATH}'")

    return vector_store


def load_vector_store():
    print("Loading existing FAISS vector store...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Vector store loaded successfully!")

    return vector_store


if __name__ == "__main__":
    create_vector_store()