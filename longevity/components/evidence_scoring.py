# longevity/components/evidence_scoring.py
import math
import json
from collections import Counter
from longevity.constants.aging_hallmarks import AGING_HALLMARKS, LONGEVITY_TARGETS


def score_by_bradford_hill_aging(evidence_list):
    """
    Score evidence using Bradford Hill criteria adapted for aging research.

    Args:
        evidence_list: List of dicts with keys: 'text', 'entities', 'pmid'

    Returns:
        dict: Scores for each criterion + composite score
    """
    scores = {}
    texts = [e.get("text", "").lower() for e in evidence_list]

    # Strength: Number of supporting papers
    scores["strength"] = min(1.0, len(texts) / 5.0)

    # Consistency: Diversity of findings
    unique_texts = len(set(texts))
    scores["consistency"] = min(1.0, unique_texts / 5.0)

    # Specificity: Presence of aging-specific mechanisms
    aging_mechanism_words = [
        kw for hallmark in AGING_HALLMARKS.values() for kw in hallmark["keywords"]
    ]
    mechanism_count = sum(
        any(word in text for word in aging_mechanism_words) for text in texts
    )
    scores["specificity"] = min(1.0, mechanism_count / 3.0)

    # Temporality: Longitudinal/temporal evidence
    temporal_words = [
        "longitudinal",
        "lifespan",
        "aging",
        "old",
        "young",
        "time-course",
        "progression",
        "age-related",
    ]
    temporal_count = sum(any(word in text for word in temporal_words) for text in texts)
    scores["temporality"] = min(1.0, temporal_count / 2.0)

    # Plausibility: Alignment with known longevity interventions
    intervention_words = [kw for target in LONGEVITY_TARGETS.values() for kw in target]
    intervention_count = sum(
        any(word in text for word in intervention_words) for text in texts
    )
    scores["plausibility"] = min(1.0, intervention_count / 2.0)

    # Coherence: Overlap with established aging hallmarks
    hallmark_coverage = sum(
        any(kw in text for kw in hallmark["keywords"])
        for text in texts
        for hallmark in AGING_HALLMARKS.values()
    )
    scores["coherence"] = min(1.0, hallmark_coverage / 5.0)

    # Reproducibility: Number of distinct PMIDs
    pmids = [e.get("pmid") for e in evidence_list if e.get("pmid")]
    scores["reproducibility"] = min(1.0, len(set(pmids)) / 3.0)

    # Biological gradient: Dose-response evidence
    dose_words = ["dose", "concentration", "gradient", "dose-dependent"]
    dose_count = sum(any(word in text for word in dose_words) for text in texts)
    scores["biological_gradient"] = min(1.0, dose_count / 2.0)

    # Composite score (weighted average)
    weights = {
        "strength": 0.15,
        "consistency": 0.15,
        "specificity": 0.15,
        "temporality": 0.10,
        "plausibility": 0.15,
        "coherence": 0.15,
        "reproducibility": 0.10,
        "biological_gradient": 0.05,
    }

    scores["composite"] = sum(scores[k] * weights[k] for k in weights.keys())

    return scores


def identify_aging_hallmarks(text):
    """Identify which aging hallmarks are mentioned in text."""
    text_lower = text.lower()
    identified = []

    for hallmark_name, hallmark_data in AGING_HALLMARKS.items():
        for keyword in hallmark_data["keywords"]:
            if keyword.lower() in text_lower:
                identified.append(
                    {
                        "hallmark": hallmark_name,
                        "keyword": keyword,
                        "description": hallmark_data["description"],
                    }
                )
                break  # Only count each hallmark once per text

    return identified


if __name__ == "__main__":
    # Demo with aging-relevant examples
    example = [
        {
            "text": "Rapamycin treatment extends lifespan in mice by inhibiting mTOR and reducing cellular senescence (p<0.001)",
            "pmid": "12345",
        },
        {
            "text": "Longitudinal study shows NAD+ supplementation improves mitochondrial function in aged humans",
            "pmid": "67890",
        },
        {
            "text": "Senolytic drug combination eliminates senescent cells and extends healthspan in naturally aged mice",
            "pmid": "11223",
        },
    ]

    scores = score_by_bradford_hill_aging(example)
    print("Bradford Hill Scores (Aging-Adapted):")
    for criterion, score in scores.items():
        print(f"  {criterion}: {score:.3f}")

    print("\n" + "=" * 50)
    print("Identified Aging Hallmarks:")
    for ex in example:
        hallmarks = identify_aging_hallmarks(ex["text"])
        print(f"\nText: {ex['text'][:80]}...")
        for h in hallmarks:
            print(f"  - {h['hallmark']}: {h['description']}")
