from app.embeddings import generate_embeddings
from app.vector_store import VectorStore


# Sample chunks
sample_chunks = [
    {
        "chunk_id": 1,
        "page": 1,
        "text": "Organizational change refers to the process of modifying an organization."
    },
    {
        "chunk_id": 2,
        "page": 2,
        "text": "Resistance to change can be reduced through communication and employee participation."
    },
    {
        "chunk_id": 3,
        "page": 3,
        "text": "Work stress can affect employee productivity, health, and overall performance."
    }
]


# Generate embeddings
embedded_chunks = generate_embeddings(
    sample_chunks
)


# Create FAISS vector store
vector_store = VectorStore()


# Add embeddings to FAISS
vector_store.add_chunks(
    embedded_chunks
)


print(
    f"FAISS vectors stored: {vector_store.index.ntotal}"
)


# Create embedding for user query
query = "What is resistance to organizational change?"

query_embedding = generate_embeddings([
    {
        "chunk_id": 0,
        "page": 0,
        "text": query
    }
])[0]["embedding"]


# Search FAISS
results = vector_store.search(
    query_embedding,
    top_k=2
)


print("\nSearch Results:")

for result in results:

    print("\n--------------------")
    print(f"Chunk ID : {result['chunk_id']}")
    print(f"Page    : {result['page']}")
    print(f"Score   : {result['score']:.4f}")
    print(f"Text    : {result['text']}")