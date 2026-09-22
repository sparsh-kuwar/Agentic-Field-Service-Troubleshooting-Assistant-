from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


PDF_PATH = "data/manuals/X200_Industrial_Water_Pump_Manual.pdf"


def create_chunks():

    # Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Split PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


def create_embeddings():

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded")

    return embeddings


if __name__ == "__main__":

    chunks = create_chunks()

    embeddings = create_embeddings()

    # Generate embedding for first chunk
    vector = embeddings.embed_query(chunks[0].page_content)

    print("\nFirst chunk:")
    print(chunks[0].page_content)

    print("\nEmbedding:")
    print(vector[:10])

    print("\nVector dimensions:")
    print(len(vector))