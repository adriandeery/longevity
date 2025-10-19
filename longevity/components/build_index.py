# scripts/build_index.py
import argparse, json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="inp", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
args = parser.parse_args()

model = SentenceTransformer(args.model)

texts = []
docs = []
with open(args.inp, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        texts.append(rec.get("text", ""))
        docs.append(rec)

embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
dim = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)
faiss.write_index(index, args.out)
# save docs
import pickle

with open(args.out + ".docs.pkl", "wb") as f:
    pickle.dump(docs, f)
print("Built FAISS index with", len(texts), "docs")
