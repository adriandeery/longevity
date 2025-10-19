# longevity/utils/rag.py
import faiss
import pickle
import numpy as np
from longevity.utils.nlp import embed_texts


class SimpleRAG:
    def __init__(self, index_path):
        self.index = faiss.read_index(index_path)
        with open(index_path + ".docs.pkl", "rb") as f:
            self.docs = pickle.load(f)

    def retrieve(self, query, k=5):
        q_emb = embed_texts([query])[0].astype("float32")
        D, I = self.index.search(q_emb.reshape(1, -1), k)
        return [self.docs[i] for i in I[0] if i < len(self.docs)]

    def retrieve_with_scores(self, query, k=5):
        """Return documents with similarity scores."""
        q_emb = embed_texts([query])[0].astype("float32")
        D, I = self.index.search(q_emb.reshape(1, -1), k)
        results = []
        for distance, idx in zip(D[0], I[0]):
            if idx < len(self.docs):
                doc = self.docs[idx].copy()
                doc["similarity_score"] = float(
                    1 / (1 + distance)
                )  # Convert distance to similarity
                results.append(doc)
        return results
