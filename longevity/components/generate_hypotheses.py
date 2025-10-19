# scripts/generate_hypotheses.py
import argparse, pickle, json
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--query", required=True)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--index", default="data/index/faiss_index.faiss")
parser.add_argument("--model", default="distilgpt2")
args = parser.parse_args()

# Load index and docs
index = faiss.read_index(args.index)
import pickle
docs = pickle.load(open(args.index + ".docs.pkl","rb"))

embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
q_emb = embed_model.encode([args.query])[0].astype("float32")
D, I = index.search(q_emb.reshape(1,-1), args.k)
retrieved = [docs[i] for i in I[0]]

# build context
context = "\n\n".join([f"TITLE: {d.get('title','')[:200]}\nABSTRACT: {d.get('abstract','')[:800]}" for d in retrieved])

prompt = f"You are a biomedical research assistant. Given the following retrieved evidence from PubMed, propose up to 3 concise, testable hypotheses relating to the query: '{args.query}'. For each hypothesis, list supporting evidence (PMID title/abstract snippets) and a short rationale.\n\nEVIDENCE:\n{context}\n\nHYPOTHESES:\n"

# generator
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model)
gen = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
out = gen(prompt, max_length=512, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]["generated_text"]
print(out)
