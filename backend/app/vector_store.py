
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add_chunks(self, embedded_chunks: list[dict]):
        if not embedded_chunks:
            return

        embeddings = np.array(
            [chunk["embedding"] for chunk in embedded_chunks],
            dtype="float32"
        )

        self.index.add(embeddings)

        for chunk in embedded_chunks:
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "document_id": chunk.get("document_id"),
                "database_chunk_id": chunk.get("database_chunk_id")
            }
            self.chunks.append(metadata)

    def search(
        self,
        query_embedding,
        top_k: int = 3,
        document_id: int | None = None
    ):
        if self.index.ntotal == 0:
            return []

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        # Search all vectors so filtering doesn't miss
        # matching chunks from the selected document.
        scores, indices = self.index.search(
            query_vector,
            self.index.ntotal
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            chunk = self.chunks[index]

            if (
                document_id is not None
                and chunk.get("document_id") != document_id
            ):
                continue

            results.append({
                **chunk,
                "score": float(score)
            })

            if len(results) >= top_k:
                break

        return results
