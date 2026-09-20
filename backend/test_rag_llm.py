import sys

from app.ingestion.router import extract_content
from app.chunking import chunk_pages
from app.embeddings import generate_embeddings
from app.vector_store import VectorStore
from app.rag import answer_question


if len(sys.argv) < 2:

    print("Please provide the PDF file path.")

    sys.exit(1)


pdf_path = sys.argv[1]


with open(pdf_path, "rb") as file:

    file_content = file.read()


print("PDF loaded successfully.")


extracted_content = extract_content(
    file_content,
    ".pdf"
)


pages = extracted_content["pages"]


print(
    f"Pages extracted: {len(pages)}"
)


chunks = chunk_pages(
    pages,
    chunk_size=800,
    chunk_overlap=100
)


print(
    f"Total chunks created: {len(chunks)}"
)


embedded_chunks = generate_embeddings(
    chunks
)


print(
    f"Total embeddings generated: "
    f"{len(embedded_chunks)}"
)


vector_store = VectorStore()


vector_store.add_chunks(
    embedded_chunks
)


print(
    f"FAISS vectors stored: "
    f"{vector_store.index.ntotal}"
)


query = input(
    "\nAsk a question about the PDF: "
)


result = answer_question(
    query,
    vector_store,
    top_k=3
)


print("\n========== ANSWER ==========\n")

print(
    result["answer"]
)


print(
    "\n========== SOURCES ==========\n"
)


for source in result["results"]:

    print(
        f"Page   : {source['page']}"
    )

    print(
        f"Score  : {source['score']:.4f}"
    )

    print(
        f"Chunk  : {source['chunk_id']}"
    )

    print("-----------------------------")