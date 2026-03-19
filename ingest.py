import fitz
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Returns a list of pages with text and metadata.
    The metadata is important—we'll show the source to the user later.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # "text" = pure text without markdown

        # Filter pages with junk (< 50 characters - most likely a header or footer)
        if len(text.strip()) < 50:
            continue

        pages.append({
            "text": text,
            "metadata": {
                "source": Path(pdf_path).name,
                "page": page_num + 1,
            }
        })

    doc.close()
    print(f"Parsed {len(pages)} pages from {pdf_path}")
    return pages

def chunk_pages(pages: list[dict]) -> list[Document]:
    """
    RecursiveCharacterTextSplitter tries to split by paragraph, then by sentence,
    then by word, preserving semantic boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,        # ~512 tokens per chunk
        chunk_overlap=50,      # overlapping to avoid losing context at the boundaries
        separators=["\n\n", "\n", ". ", " "],  # order of priority of breakdown
    )

    documents = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    **page["metadata"],
                    "chunk_index": i,
                }
            ))

    print(f"Created {len(documents)} chunks")
    return documents

def build_vectorstore(documents: list[Document], persist_dir: str = "./chroma_db") -> Chroma:
    """
    HuggingFaceEmbeddings runs the model locally.
    all-MiniLM-L6-v2 is small (80MB), fast, and a good quality starter.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # "cuda" if GPU
    )

    # Chroma saves to disk - no need to reindex on next launch
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="legal_docs",
    )

    print(f"Vectorstore built and saved to {persist_dir}")
    return vectorstore


# Entry point for indexing
if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"

    pages = parse_pdf(pdf_path)
    documents = chunk_pages(pages)
    vectorstore = build_vectorstore(documents)

    print("Ingestion complete. Ready to query.")