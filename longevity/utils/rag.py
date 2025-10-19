# utils/rag.py
import faiss
import pickle
from utils.nlp import embed_texts

class SimpleRAG:
    def __init__(self, index_path):
        self.index = faiss.read_index(index_path)
        self.docs = pickle.load(open(index_path + ".docs.pkl","rb"))

    def retrieve(self, query, k=5):
        q_emb = embed_texts([query])[0].astype('float32')
        D,I = self.index.search(q_emb.reshape(1,-1), k)
        return [self.docs[i] for i in I[0]]
