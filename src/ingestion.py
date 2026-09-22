from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "data/manuals/X200_Industrial_Water_Pump_Manual.pdf"


def load_and_split_manual():

    # Step 1: Load the PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    # Step 2: Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
    # Step 3: Split documents into chunks
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # Step 4: Display some chunks
    for i, chunk in enumerate(chunks[:5]):
        print("\n==============================")
        print(f"CHUNK {i + 1}")
        print("==============================")
        print(chunk.page_content)
        print("\nMetadata:")
        print(chunk.metadata)

    return chunks


if __name__ == "__main__":
    load_and_split_manual()