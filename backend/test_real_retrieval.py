import sys

from app.ingestion.router import extract_content
from app.chunking import chunk_pages
from app.embeddings import generate_embeddings
from app.vector_store import VectorStore


if len(sys.argv) < 2:
    print("Please provide the PDF file path.")
    print(
        r'Example: python test_real_retrieval.py "C:\path\Document 6.pdf"'
    )
    sys.exit(1)


pdf_path = sys.argv[1]


# --------------------------------------------------
# 1. Read PDF
# --------------------------------------------------

with open(pdf_path, "rb") as file:
    file_content = file.read()


print("PDF loaded successfully.")


# --------------------------------------------------
# 2. Extract content
# --------------------------------------------------

extracted_content = extract_content(
    file_content,
    ".pdf"
)


pages = extracted_content["pages"]

print(
    f"Pages extracted: {len(pages)}"
)


# --------------------------------------------------
# 3. Create chunks
# --------------------------------------------------

chunks = chunk_pages(
    pages,
    chunk_size=800,
    chunk_overlap=100
)


print(
    f"Total chunks created: {len(chunks)}"
)


# --------------------------------------------------
# 4. Generate embeddings
# --------------------------------------------------

embedded_chunks = generate_embeddings(
    chunks
)


print(
    f"Total embeddings generated: "
    f"{len(embedded_chunks)}"
)


# --------------------------------------------------
# 5. Create FAISS vector store
# --------------------------------------------------

vector_store = VectorStore()

vector_store.add_chunks(
    embedded_chunks
)


print(
    f"FAISS vectors stored: "
    f"{vector_store.index.ntotal}"
)


# --------------------------------------------------
# 6. Ask a question
# --------------------------------------------------

query = input(
    "\nAsk a question about the PDF: "
)


query_embedding = generate_embeddings([
    {
        "chunk_id": 0,
        "page": 0,
        "text": query
    }
])[0]["embedding"]


# --------------------------------------------------
# 7. Search relevant chunks
# --------------------------------------------------

results = vector_store.search(
    query_embedding,
    top_k=3
)


print("\n========== SEARCH RESULTS ==========")


for result in results:

    print("\n--------------------")

    print(
        f"Chunk ID : {result['chunk_id']}"
    )

    print(
        f"Page    : {result['page']}"
    )

    print(
        f"Score   : {result['score']:.4f}"
    )

    print(
        f"Text    : {result['text']}"
    )