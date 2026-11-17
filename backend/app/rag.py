
from app.embeddings import generate_embeddings
from app.vector_store import VectorStore
from app.llm import generate_answer


# --------------------------------------------------
# RETRIEVE RELEVANT CONTEXT
# --------------------------------------------------

def retrieve_context(
    query: str,
    vector_store: VectorStore,
    top_k: int = 3,
    document_id: int | None = None
) -> dict:

    if not query.strip():
        return {
            "query": query,
            "results": [],
            "context": ""
        }

    # Generate embedding for the user's query
    query_embedding = generate_embeddings([
        {
            "chunk_id": 0,
            "page": 0,
            "text": query
        }
    ])[0]["embedding"]

    # Retrieve relevant chunks
    results = vector_store.search(
        query_embedding,
        top_k=top_k,
        document_id=document_id
    )

    # Prepare context with source information
    context_parts = []

    for result in results:

        chunk_id = result.get("database_chunk_id")
        page_number = result["page"]
        chunk_text = result["text"]

        context_parts.append(
            f"[Chunk ID: {chunk_id} | Page: {page_number}]\n"
            f"{chunk_text}"
        )

    context = "\n\n".join(context_parts)

    return {
        "query": query,
        "results": results,
        "context": context
    }


# --------------------------------------------------
# GENERATE ANSWER USING RAG
# --------------------------------------------------

def answer_question(
    query: str,
    vector_store: VectorStore,
    top_k: int = 3,
    document_id: int | None = None
) -> dict:

    retrieval_result = retrieve_context(
        query=query,
        vector_store=vector_store,
        top_k=top_k,
        document_id=document_id
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

    # Generate answer using Gemini
    answer = generate_answer(
        question=query,
        context=context
    )

    return {
        **retrieval_result,
        "answer": answer
    }
