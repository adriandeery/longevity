# Longevity Pipeline
This repository is a runnable prototype for a biomedical hypothesis-generation pipeline.
It's designed for development on Windows/VSCode and to be compatible with a modest GPU (3GB)
by using parameter-efficient fine-tuning (PEFT/LoRA) and small models locally, while offloading heavy tasks
(AlphaFold/large fine-tuning/docking) to the cloud.

# Structure
---------
- longevity/components/
    - fetch_pubmed.py         : fetch PubMed abstracts via Entrez
    - preprocess.py          : scispacy-based entity extraction and text cleaning
    - build_index.py         : create embeddings and a FAISS index
    - fine_tune_lora.py      : example LoRA fine-tune config and training loop (small)
    - generate_hypotheses.py : RAG-style retrieval + generation wrapper
    - evidence_scoring.py    : scoring module implementing Bradford Hill-like checklist
    - esmfold_stub.py        : stub + instructions to run ESMFold/ColabFold (external)
- app/
    - api.py                 : FastAPI demo showing how to query the model
- longevity/utils/
    - nlp.py                 : shared NLP helpers
    - rag.py                 : retrieval-augmented generation helpers


Quick start (development)
-------------------------
1. Create a conda environment:
    conda create -p ./longevity-pipeline python==3.10
    conda activate ./longevity-pipeline

    # Optional: If you have gpu. Also run nvidia-smi to find which CUDA version you have e.g. 12.8
    conda install cudatoolkit=12.8 -c conda-forge

2. Install required packages:
   pip install -r requirements.txt
   NOTE: scispacy requires model install: python -m spacy download en_core_sci_sm

3. Set Entrez email in environment variable:
   set ENTRZ_EMAIL=you@example.com   (Windows CMD)
   or
   $env:ENTRZ_EMAIL="you@example.com" (PowerShell)

4. Fetch sample PubMed data:
   python scripts/fetch_pubmed.py --query "hallmarks of aging" --n 50 --out data/pubmed/raw.jsonl

5. Preprocess and build index:
   python scripts/preprocess.py --in data/pubmed/raw.jsonl --out data/pubmed/processed.jsonl
   python scripts/build_index.py --in data/pubmed/processed.jsonl --out data/index/faiss_index.faiss

6. Fine-tune (optional, small prototype):
   python scripts/fine_tune_lora.py --data data/pubmed/for_finetune.jsonl --output ./models/lora_biomed

7. Run generation demo:
   python scripts/generate_hypotheses.py --query "inhibit telomerase effect on senescence" --k 5

8. Run API:
   uvicorn app.api:app --reload

Notes on heavy tasks
--------------------
- AlphaFold / full ESMFold runs are heavy. See scripts/esmfold_stub.py for instructions to run via ColabFold or cloud GPUs.
- For large LLM fine-tuning (>1B params) use cloud GPUs or Hugging Face Training on a GPU instance.

Limitations
-----------
- This is a prototype. Outputs are hypotheses, not validated biological claims.
- Use appropriate biosafety, ethics review, and domain expert oversight before any experimental follow-up.


