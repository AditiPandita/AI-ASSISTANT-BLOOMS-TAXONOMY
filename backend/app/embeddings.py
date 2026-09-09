from sentence_transformers import SentenceTransformer


# Load the embedding model once
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def generate_embeddings(
    chunks: list[dict]
) -> list[dict]:

    if not chunks:
        return []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embedded_chunks = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):
        embedded_chunks.append({
            **chunk,
            "embedding": embedding
        })

    return embedded_chunks