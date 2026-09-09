from app.embeddings import generate_embeddings


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
    }
]


embedded_chunks = generate_embeddings(sample_chunks)


print(f"Total embedded chunks: {len(embedded_chunks)}")

for chunk in embedded_chunks:
    print("\n--------------------")
    print(f"Chunk ID : {chunk['chunk_id']}")
    print(f"Page    : {chunk['page']}")
    print(f"Text    : {chunk['text']}")
    print(f"Vector dimensions : {len(chunk['embedding'])}")
    print(f"First 5 values : {chunk['embedding'][:5]}")