from app.embeddings import generate_embeddings
from app.vector_store import VectorStore


def retrieve_context(
    query: str,
    vector_store: VectorStore,
    top_k: int = 3
) -> dict:

    if not query.strip():
        return {
            "query": query,
            "results": [],
            "context": ""
        }

    # Convert user question into an embedding
    query_embedding = generate_embeddings([
        {
            "chunk_id": 0,
            "page": 0,
            "text": query
        }
    ])[0]["embedding"]

    # Search FAISS for relevant chunks
    results = vector_store.search(
        query_embedding,
        top_k=top_k
    )

    # Build context for the LLM
    context_parts = []

    for result in results:

        context_parts.append(
            f"[Page {result['page']}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    return {
        "query": query,
        "results": results,
        "context": context
    }