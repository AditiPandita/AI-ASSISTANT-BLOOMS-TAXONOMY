import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int = 384):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.chunks = []


    def add_chunks(
        self,
        embedded_chunks: list[dict]
    ):

        if not embedded_chunks:
            return

        embeddings = np.array(
            [
                chunk["embedding"]
                for chunk in embedded_chunks
            ],
            dtype="float32"
        )

        self.index.add(embeddings)

        self.chunks.extend(
            [
                {
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                for chunk in embedded_chunks
            ]
        )


    def search(
        self,
        query_embedding,
        top_k: int = 3
    ):

        if self.index.ntotal == 0:
            return []

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append({
                **self.chunks[index],
                "score": float(score)
            })

        return results