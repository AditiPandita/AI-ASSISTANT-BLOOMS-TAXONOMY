import sys

from app.ingestion.router import extract_content
from app.chunking import chunk_pages
from app.embeddings import generate_embeddings
from app.vector_store import VectorStore
from app.rag import retrieve_context


if len(sys.argv) < 2:
    print("Please provide the PDF file path.")
    sys.exit(1)


pdf_path = sys.argv[1]


# 1. Load PDF
with open(pdf_path, "rb") as file:
    file_content = file.read()

print("PDF loaded successfully.")


# 2. Extract text
extracted_content = extract_content(
    file_content,
    ".pdf"
)

pages = extracted_content["pages"]

print(
    f"Pages extracted: {len(pages)}"
)


# 3. Create chunks
chunks = chunk_pages(
    pages,
    chunk_size=800,
    chunk_overlap=100
)

print(
    f"Total chunks created: {len(chunks)}"
)


# 4. Generate embeddings
embedded_chunks = generate_embeddings(
    chunks
)

print(
    f"Total embeddings generated: {len(embedded_chunks)}"
)


# 5. Create FAISS store
vector_store = VectorStore()

vector_store.add_chunks(
    embedded_chunks
)

print(
    f"FAISS vectors stored: {vector_store.index.ntotal}"
)


# 6. Ask question
query = input(
    "\nAsk a question about the PDF: "
)


# 7. Retrieve context using RAG module
rag_result = retrieve_context(
    query,
    vector_store,
    top_k=3
)


# 8. Display results
print("\n========== RETRIEVED CONTEXT ==========")

print(
    rag_result["context"]
)


print("\n========== SOURCE DETAILS ==========")

for result in rag_result["results"]:

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