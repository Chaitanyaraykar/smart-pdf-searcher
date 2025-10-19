# app/retriever.py

import faiss
import numpy as np
import pickle
from typing import List, Tuple


class VectorStore:
    def __init__(self, dim: int, store_path: str = "vector_store.faiss"):
        self.dim = dim
        self.store_path = store_path
        self.index = faiss.IndexFlatL2(dim)  # L2 distance index
        self.text_chunks = []  # to store the actual text

    # -----------------------------
    # Add vectors to FAISS index
    # -----------------------------
    def add_embeddings(self, embeddings: List[List[float]], chunks: List[str]):
        np_embeddings = np.array(embeddings).astype("float32")
        self.index.add(np_embeddings)
        self.text_chunks.extend(chunks)

    # -----------------------------
# Save / Load index
    # -----------------------------
    def save(self):
        faiss.write_index(self.index, self.store_path)
        with open(self.store_path + ".pkl", "wb") as f:
            pickle.dump(self.text_chunks, f)
        print(f"💾 Saved index and text store → {self.store_path}")

    def load(self):
        self.index = faiss.read_index(self.store_path)
        with open(self.store_path + ".pkl", "rb") as f:
            self.text_chunks = pickle.load(f)
        print(f"📂 Loaded index with {len(self.text_chunks)} chunks")

    # -----------------------------
    # Query similar chunks
    # -----------------------------
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], float(dist)))
        return results


if __name__ == "__main__":
    # Demo with dummy embeddings
    dim = 5
    store = VectorStore(dim)

    dummy_embs = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.1, 0.4, 0.3, 0.7],
        [0.9, 0.8, 0.2, 0.1, 0.3],
    ]
    dummy_chunks = [
        "AI is revolutionizing the world.",
        "Machine learning is a subset of AI.",
        "Natural Language Processing enables chatbots."
    ]

    store.add_embeddings(dummy_embs, dummy_chunks)
    query_emb = [0.15, 0.2, 0.35, 0.4, 0.6]

    results = store.search(query_emb)
    for text, score in results:
        print(f"🔍 {score:.4f} → {text}")

