# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from utils.rag import SimpleRAG
import subprocess, json
app = FastAPI(title="Biomed AI Pipeline Demo")

rag = SimpleRAG("data/index/faiss_index.faiss")

class Query(BaseModel):
    query: str

@app.post("/hypotheses")
async def get_hypotheses(q: Query):
    # For demo, call the generate script and return text (in production, integrate directly)
    import sys, io
    from scripts import generate_hypotheses as genmod  # local import for conceptual demo
    # We'll emulate by returning retrieved docs + simple prompt template
    docs = rag.retrieve(q.query, k=3)
    items = [{"title":d.get("title"), "abstract": d.get("abstract")[:500]} for d in docs]
    return {"query": q.query, "retrieved": items}
