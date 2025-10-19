# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from longevity.utils.rag import SimpleRAG
from longevity.components.generate_hypotheses import generate_hypotheses
from longevity.components.evidence_scoring import score_by_bradford_hill_aging
import os
from pathlib import Path

app = FastAPI(
    title="Longevity Hypothesis Generation API",
    description="Generate and score hypotheses for aging interventions",
    version="0.1.0",
)

# Initialize RAG system
INDEX_PATH = Path("data/index/faiss_index.faiss")
if not INDEX_PATH.exists():
    raise RuntimeError(
        f"FAISS index not found at {INDEX_PATH}. Run build_index.py first."
    )

rag = SimpleRAG(str(INDEX_PATH))


class HypothesisQuery(BaseModel):
    query: str
    top_k: int = 5
    generate_text: bool = True


class HypothesisResponse(BaseModel):
    query: str
    retrieved_papers: list
    generated_hypothesis: str | None
    evidence_scores: dict | None


@app.get("/health")
async def health_check():
    return {"status": "healthy", "index_loaded": rag.index is not None}


@app.post("/hypotheses", response_model=HypothesisResponse)
async def get_hypotheses(q: HypothesisQuery):
    """
    Generate hypotheses based on retrieved scientific literature.
    """
    try:
        # Retrieve relevant documents
        docs = rag.retrieve(q.query, k=q.top_k)

        retrieved_papers = [
            {
                "title": d.get("title", "")[:200],
                "abstract": d.get("abstract", "")[:500],
                "pmid": d.get("pmid", "N/A"),
            }
            for d in docs
        ]

        # Optionally generate hypothesis text
        hypothesis_text = None
        scores = None

        if q.generate_text:
            # This would call your generation model
            # hypothesis_text = generate_hypotheses(q.query, docs)
            hypothesis_text = "[Placeholder: Model generation would go here]"

            # Score the evidence
            evidence_list = [
                {"text": d.get("abstract", ""), "pmid": d.get("pmid")} for d in docs
            ]
            scores = score_by_bradford_hill_aging(evidence_list)

        return HypothesisResponse(
            query=q.query,
            retrieved_papers=retrieved_papers,
            generated_hypothesis=hypothesis_text,
            evidence_scores=scores,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/aging-hallmarks")
async def get_aging_hallmarks():
    """Return the 12 hallmarks of aging."""
    from longevity.constants.aging_hallmarks import AGING_HALLMARKS

    return {"hallmarks": AGING_HALLMARKS}
