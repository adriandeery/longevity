# scripts/evidence_scoring.py
import math
import json
from collections import Counter

def score_by_bradford_hill(evidence_list):
    # evidence_list: list of dicts with keys: 'text', 'entities', 'pmid' (optional)
    # This is a simple heuristic scoring mapping to Bradford Hill aspects.
    scores = {}
    texts = [e.get('text','') for e in evidence_list]
    # Strength: proxy by number of papers / mentions
    scores['strength'] = min(1.0, len(texts)/5.0)
    # Consistency: diversity of different phrases / findings
    uniq = len(set(texts))
    scores['consistency'] = min(1.0, uniq/5.0)
    # Specificity: presence of clear mechanism words
    mech_words = ['pathway','activate','inhibit','phosphorylation','binds','cleavage','ubiquitin']
    mech_count = sum(any(w in t.lower() for w in mech_words) for t in texts)
    scores['specificity'] = min(1.0, mech_count/3.0)
    # Temporality: presence of longitudinal/temporal words
    temp_words = ['time','follow-up','longitudinal','before','after','progression']
    temp = sum(any(w in t.lower() for w in temp_words) for t in texts)
    scores['temporality'] = min(1.0, temp/2.0)
    # Plausibility: placeholder (structure/docking result can increase this)
    scores['plausibility'] = 0.0
    # Coherence: overlap with known cancer hallmarks (heuristic)
    hallmark_words = ['proliferation','apoptosis','senescence','angiogenesis','metastasis','metabolic']
    hall = sum(any(w in t.lower() for w in hallmark_words) for t in texts)
    scores['coherence'] = min(1.0, hall/3.0)
    # Reproducibility proxy: number of distinct PMIDs (if provided)
    pmids = [e.get('pmid') for e in evidence_list if e.get('pmid')]
    scores['reproducibility'] = min(1.0, len(set(pmids))/3.0)
    # Composite (simple average)
    scores['composite'] = sum(scores.values())/len(scores)
    return scores

if __name__ == '__main__':
    # demo
    example = [{"text":"In mouse model, inhibition of X reduces proliferation (p<0.01)","pmid":"123"},
               {"text":"Cell study shows X binds Y and triggers apoptosis","pmid":"456"}]
    print(score_by_bradford_hill(example))
