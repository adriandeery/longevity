# longevity/components/generate_hypotheses.py
import argparse
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from longevity.constants.aging_hallmarks import AGING_HALLMARKS, LONGEVITY_TARGETS
from longevity.logging.logger import logger
from longevity.exception.exception import ModelError
import sys


def load_rag_system(index_path):
    """Load FAISS index and documents."""
    try:
        index = faiss.read_index(index_path)
        with open(index_path + ".docs.pkl", "rb") as f:
            docs = pickle.load(f)
        logger.info(f"Loaded index with {index.ntotal} documents")
        return index, docs
    except Exception as e:
        raise ModelError(f"Failed to load RAG system: {str(e)}", sys)


def retrieve_documents(query, index, docs, embed_model, k=5):
    """Retrieve top-k relevant documents."""
    q_emb = embed_model.encode([query])[0].astype("float32")
    D, I = index.search(q_emb.reshape(1, -1), k)
    retrieved = [docs[i] for i in I[0] if i < len(docs)]
    logger.info(f"Retrieved {len(retrieved)} documents for query: {query[:50]}...")
    return retrieved, D[0], I[0]


def build_structured_prompt(query, retrieved_docs, aging_hallmarks=True):
    """
    Build a structured prompt for hypothesis generation.

    Uses chain-of-thought prompting and focuses on aging biology.
    """
    # Extract evidence
    evidence_snippets = []
    for i, doc in enumerate(retrieved_docs[:3], 1):
        title = doc.get("title", "No title")[:150]
        abstract = doc.get("abstract", "No abstract")[:400]
        evidence_snippets.append(
            f"[Paper {i}]\nTitle: {title}\nKey Finding: {abstract}"
        )

    evidence_text = "\n\n".join(evidence_snippets)

    # Build aging-focused prompt
    prompt = f"""You are an expert in aging biology and longevity research. 

QUERY: {query}

RELEVANT SCIENTIFIC EVIDENCE:
{evidence_text}

TASK: Generate 2-3 testable hypotheses that:
1. Target one or more hallmarks of aging (genomic instability, telomere attrition, epigenetic alterations, loss of proteostasis, disabled macroautophagy, deregulated nutrient sensing, mitochondrial dysfunction, cellular senescence, stem cell exhaustion, altered intercellular communication, chronic inflammation, dysbiosis)
2. Propose a specific molecular mechanism
3. Suggest an experimental validation approach
4. Consider translatability to human longevity

HYPOTHESES:
"""

    return prompt


def generate_hypotheses_structured(
    query,
    index_path="data/index/faiss_index.faiss",
    model_name="google/flan-t5-base",  # Better than distilgpt2
    k=5,
    device=None,
):
    """
    Generate hypotheses using improved prompting and better models.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    logger.info(f"Starting hypothesis generation for: {query}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Device: {'CUDA' if device >= 0 else 'CPU'}")

    # Load RAG system
    index, docs = load_rag_system(index_path)
    embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Retrieve documents
    retrieved, distances, indices = retrieve_documents(
        query, index, docs, embed_model, k=k
    )

    # Build prompt
    prompt = build_structured_prompt(query, retrieved)

    logger.info(f"Prompt length: {len(prompt)} characters")

    # Generate with better model (Flan-T5 is instruction-tuned)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device >= 0:
            model = model.to("cuda")

        # For Flan-T5, we can use generate() directly
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        )

        if device >= 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate with improved parameters
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,  # Beam search for better quality
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            early_stopping=True,
            num_return_sequences=1,
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info("Successfully generated hypotheses")

        # Format output
        result = {
            "query": query,
            "generated_hypotheses": generated_text,
            "supporting_evidence": [
                {
                    "title": doc.get("title", "")[:200],
                    "abstract": doc.get("abstract", "")[:300],
                    "pmid": doc.get("pmid", "N/A"),
                    "relevance_score": float(1 / (1 + dist)),
                }
                for doc, dist in zip(retrieved, distances)
            ],
            "model_used": model_name,
            "timestamp": str(Path(__file__).stat().st_mtime),
        }

        return result

    except Exception as e:
        raise ModelError(f"Failed to generate hypotheses: {str(e)}", sys)


def format_output_for_display(result):
    """Pretty print the results."""
    print("\n" + "=" * 80)
    print("LONGEVITY HYPOTHESIS GENERATION RESULTS")
    print("=" * 80)
    print(f"\nQuery: {result['query']}")
    print(f"Model: {result['model_used']}")
    print("\n" + "-" * 80)
    print("GENERATED HYPOTHESES:")
    print("-" * 80)
    print(result["generated_hypotheses"])
    print("\n" + "-" * 80)
    print("SUPPORTING EVIDENCE:")
    print("-" * 80)
    for i, evidence in enumerate(result["supporting_evidence"], 1):
        print(f"\n[{i}] {evidence['title']}")
        print(
            f"    PMID: {evidence['pmid']} | Relevance: {evidence['relevance_score']:.3f}"
        )
        print(f"    {evidence['abstract'][:200]}...")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate longevity research hypotheses using RAG"
    )
    parser.add_argument("--query", required=True, help="Research question")
    parser.add_argument(
        "--k", type=int, default=5, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--index", default="data/index/faiss_index.faiss", help="Path to FAISS index"
    )
    parser.add_argument(
        "--model",
        default="google/flan-t5-base",
        help="Model name (flan-t5-base, flan-t5-large, distilgpt2)",
    )
    parser.add_argument(
        "--output", default=None, help="Optional: save results to JSON file"
    )

    args = parser.parse_args()

    # Generate hypotheses
    result = generate_hypotheses_structured(
        query=args.query, index_path=args.index, model_name=args.model, k=args.k
    )

    # Display results
    format_output_for_display(result)

    # Optionally save to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
