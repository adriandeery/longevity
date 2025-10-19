# utils/nlp.py
from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed_texts(texts):
    return embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
