# Longevity Hypothesis Generation Pipeline

An AI-powered pipeline for generating and scoring testable hypotheses targeting the **12 hallmarks of aging** using foundation models, retrieval-augmented generation (RAG), and evidence-based scoring.

## 🧬 Features

- **Literature Mining**: Automated PubMed abstract retrieval
- **Semantic Search**: FAISS-based vector similarity search
- **Hypothesis Generation**: LLM-powered hypothesis formulation focused on longevity
- **Evidence Scoring**: Bradford Hill criteria adapted for aging research
- **Hallmark Identification**: Automatic mapping to 12 hallmarks of aging
- **API Interface**: FastAPI REST API for integration
- **Modular Design**: Clean ML engineering architecture

## 📋 Hallmarks of Aging (López-Otín et al., 2023)

This pipeline targets interventions for the 12 established hallmarks:

1. Genomic instability
2. Telomere attrition
3. Epigenetic alterations
4. Loss of proteostasis
5. Disabled macroautophagy
6. Deregulated nutrient sensing
7. Mitochondrial dysfunction
8. Cellular senescence
9. Stem cell exhaustion
10. Altered intercellular communication
11. Chronic inflammation
12. Dysbiosis

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **CUDA 12.1+** (optional, for GPU acceleration)
- **4GB+ GPU** (optional, pipeline works on CPU)
- **Windows 11 / Linux / macOS**

### Installation
```bash
# 1. Clone repository
git clone <your-repo-url>
cd longevity-pipeline

# 2. Create conda environment
conda create -n longevity python=3.10 -y
conda activate longevity

# 3. Install PyTorch with CUDA (if you have GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install requirements
pip install -r requirements.txt

# 5. Download scispacy model
python -m spacy download en_core_sci_sm

# 6. Set environment variables
cp .env.example .env
# Edit .env and add your email for NCBI Entrez
```

### Environment Setup (.env file)
```bash
# Required for PubMed API
ENTREZ_EMAIL=your.email@example.com

# Optional: Model configurations
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
GENERATION_MODEL=google/flan-t5-base
```

### Run Complete Pipeline
```bash
# Run everything from scratch
python scripts/run_pipeline.py --full

# Or run specific stages:
python scripts/run_pipeline.py \
  --fetch-data \
  --query "rapamycin longevity mechanisms" \
  --num-papers 200 \
  --preprocess \
  --build-index \
  --generate \
  --hypothesis-query "Can rapamycin extend human healthspan?"
```

### Generate Hypotheses Only
```bash
# Assumes you already have indexed data
python -m longevity.pipeline.inference_pipeline \
  --query "Do senolytics improve cognitive function in aged individuals?" \
  --top-k 10 \
  --output outputs/senolytics_hypothesis.json
```

### Run API Server
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST "http://localhost:8000/hypotheses" \
  -H "Content-Type: application/json" \
  -d '{"query": "NAD+ boosters and mitochondrial function", "top_k": 5}'
```

## 📁 Project Structure
```
longevity-pipeline/
├── longevity/
│   ├── components/          # Core pipeline components
│   │   ├── fetch_pubmed.py
│   │   ├── preprocess.py
│   │   ├── build_index.py
│   │   ├── generate_hypotheses.py
│   │   └── evidence_scoring.py
│   ├── pipeline/            # Orchestration
│   │   └── inference_pipeline.py
│   ├── config/              # Configuration management
│   │   └── settings.py
│   ├── constants/           # Aging hallmarks, targets
│   │   └── aging_hallmarks.py
│   ├── utils/               # Shared utilities
│   │   ├── nlp.py
│   │   └── rag.py
│   ├── logging/             # Structured logging
│   └── exception/           # Custom exceptions
├── app/
│   └── api.py               # FastAPI application
├── scripts/
│   └── run_pipeline.py      # Master orchestration script
├── data/                    # Data storage (gitignored)
├── models/                  # Trained models (gitignored)
├── outputs/                 # Generated hypotheses
└── requirements.txt
```

## 🔧 Configuration

Edit `longevity/config/settings.py` or use environment variables:
```python
# Model settings
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
GENERATION_MODEL=google/flan-t5-base  # or flan-t5-large for better quality

# RAG settings
TOP_K_RETRIEVAL=5

# GPU settings
USE_CUDA=true
MAX_GPU_MEMORY=4GB
```

## 🧪 Example Hypotheses
```bash
python -m longevity.pipeline.inference_pipeline \
  --query "Can combining senolytics with NAD+ boosters synergistically extend healthspan?"
```

**Output:**
```
HYPOTHESES:
1. Senolytic-mediated clearance of senescent cells reduces SASP-driven inflammation,
   which may enhance NAD+ biosynthesis and mitochondrial function...
   
EVIDENCE QUALITY (Bradford Hill):
  strength.................... 0.800
  consistency................. 0.600
  temporality................. 0.500
  coherence................... 0.900
  composite................... 0.723

IDENTIFIED HALLMARKS:
  - cellular_senescence
  - mitochondrial_dysfunction
  - chronic_inflammation
  - deregulated_nutrient_sensing
```

## 🎯 GPU Considerations

**For 4GB GPU (Your Setup):**
- Use `faiss-cpu` (already in requirements)
- Use small models: `flan-t5-base` (250M params) or `distilgpt2` (82M)
- Batch size = 1-2
- Enable gradient checkpointing if fine-tuning

**For Better Performance:**
- 8GB+ GPU: Use `flan-t5-large` (780M params)
- 16GB+ GPU: Use `BioGPT-Large` or fine-tune larger models
- Cloud: Use Google Colab/AWS for ESMFold structure prediction

## 🔬 Advanced Usage

### Fine-tune on Custom Data
```bash
# Prepare training data (prompt/completion pairs)
python scripts/prepare_training_data.py \
  --input data/pubmed/processed.jsonl \
  --output data/training/finetune.jsonl

# Fine-tune with LoRA
python -m longevity.components.fine_tune_lora \
  --data data/training/finetune.jsonl \
  --output models/lora_longevity \
  --model google/flan-t5-base
```

### Batch Processing
```python
from longevity.pipeline.inference_pipeline import LongevityInferencePipeline

pipeline = LongevityInferencePipeline()

queries = [
    "Can metformin prevent age-related cognitive decline?",
    "Do telomerase activators extend human lifespan?",
    "Can autophagy inducers delay onset of neurodegenerative diseases?"
]

results = pipeline.run_batch(queries, output_dir="outputs/batch_results")
```

## 📊 Output Format

Hypothesis generation produces structured JSON:
```json
{
  "query": "Can NAD+ boosters extend healthspan?",
  "generated_hypotheses": "...",
  "supporting_evidence": [...],
  "evidence_quality_scores": {
    "strength": 0.8,
    "consistency": 0.6,
    "composite": 0.72
  },
  "identified_hallmarks": {...},
  "metadata": {
    "model": "google/flan-t5-base",
    "pipeline_duration_seconds": 12.5,
    "timestamp": "2025-01-15T10:30:00"
  }
}
```

## ⚠️ Limitations & Disclaimers

- **Research Tool Only**: Outputs are computational hypotheses, NOT validated biological claims
- **Requires Expert Review**: All hypotheses must be reviewed by domain experts
- **No Medical Advice**: Not intended for clinical decision-making
- **Bias Warning**: Models may reflect biases in training data
- **Experimental Validation Required**: Hypotheses must be tested experimentally

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Better foundation models (BioGPT, Galactica)
- Multi-modal integration (protein structures, pathways)
- Automated experimental design suggestions
- Integration with molecular docking tools
- Better evaluation metrics

## 📚 References

- López-Otín et al. (2023). "Hallmarks of aging: An expanding universe." Cell.
- Bradford Hill criteria for causation
- RAG: Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

## 📝 License

[Your License Here]

## 🐛 Troubleshooting

**CUDA Installation Issues:**
```bash
# Don't install CUDA toolkit separately
# Use PyTorch's bundled CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Scispacy Model Not Found:**
```bash
python -m spacy download en_core_sci_sm
```

**Out of Memory (4GB GPU):**
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
```

**Import Errors:**
```bash
# Make sure you're in the project root and have installed in editable mode
pip install -e .

# The pipeline includes rate limiting, but if you hit limits:
# 1. Reduce --num-papers
# 2. Use NCBI API key (higher rate limit)
export NCBI_API_KEY=your_api_key_here

# If index is corrupted, rebuild:
rm -rf data/index/*
python -m longevity.components.build_index \
  --in data/pubmed/processed.jsonl \
  --out data/index/faiss_index.faiss