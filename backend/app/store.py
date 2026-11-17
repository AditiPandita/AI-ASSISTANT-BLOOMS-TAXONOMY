
from app.vector_store import VectorStore
from app.database.models import Chunk


# Shared FAISS vector store
vector_store = VectorStore()


def rebuild_vector_store(db):

    # Start with a fresh FAISS index
    vector_store.index.reset()
    vector_store.chunks.clear()

    # Retrieve all saved chunks from PostgreSQL
    saved_chunks = (
        db.query(Chunk)
        .order_by(Chunk.id)
        .all()
    )

    embedded_chunks = []

    for chunk in saved_chunks:

        embedded_chunks.append({
            "chunk_id": chunk.chunk_id,
            "page": chunk.page_number,
            "text": chunk.text,
            "document_id": chunk.document_id,
            "database_chunk_id": chunk.id,
            "embedding": chunk.embedding
        })

    # Rebuild FAISS index
    vector_store.add_chunks(embedded_chunks)

    return vector_store.index.ntotal
