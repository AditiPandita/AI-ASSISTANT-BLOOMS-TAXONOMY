from app.embeddings import generate_embeddings
from app.vector_store import VectorStore
from app.llm import generate_answer


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

    query_embedding = generate_embeddings([
        {
            "chunk_id": 0,
            "page": 0,
            "text": query
        }
    ])[0]["embedding"]

    results = vector_store.search(
        query_embedding,
        top_k=top_k
    )

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


def answer_question(
    query: str,
    vector_store: VectorStore,
    top_k: int = 3
) -> dict:

    retrieval_result = retrieve_context(
        query,
        vector_store,
        top_k=top_k
    )

    context = retrieval_result["context"]

    if not context:
        return {
            **retrieval_result,
            "answer": (
                "I could not find relevant information "
                "in the uploaded document."
            )
        }

    answer = generate_answer(
        question=query,
        context=context
    )

    return {
        **retrieval_result,
        "answer": answer
    }