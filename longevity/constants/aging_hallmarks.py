# longevity/constants/aging_hallmarks.py
"""
Hallmarks of Aging (López-Otín et al., 2023, Cell)
https://doi.org/10.1016/j.cell.2022.11.001
"""

AGING_HALLMARKS = {
    "genomic_instability": {
        "keywords": [
            "DNA damage",
            "mutation",
            "genome instability",
            "telomere",
            "chromosomal",
        ],
        "description": "Accumulation of genetic damage throughout life",
    },
    "telomere_attrition": {
        "keywords": [
            "telomere",
            "telomerase",
            "chromosome ends",
            "replicative senescence",
        ],
        "description": "Progressive shortening of telomeres",
    },
    "epigenetic_alterations": {
        "keywords": [
            "methylation",
            "histone",
            "chromatin",
            "epigenetic",
            "acetylation",
        ],
        "description": "Changes in DNA methylation and chromatin structure",
    },
    "loss_of_proteostasis": {
        "keywords": [
            "protein aggregation",
            "chaperone",
            "autophagy",
            "proteasome",
            "unfolded protein",
        ],
        "description": "Decline in protein quality control",
    },
    "disabled_macroautophagy": {
        "keywords": [
            "autophagy",
            "lysosome",
            "mitophagy",
            "chaperone-mediated autophagy",
        ],
        "description": "Impaired cellular recycling mechanisms",
    },
    "deregulated_nutrient_sensing": {
        "keywords": [
            "mTOR",
            "AMPK",
            "insulin",
            "IGF-1",
            "nutrient sensing",
            "caloric restriction",
        ],
        "description": "Disruption of metabolic signaling pathways",
    },
    "mitochondrial_dysfunction": {
        "keywords": [
            "mitochondria",
            "oxidative stress",
            "ROS",
            "respiratory chain",
            "ATP",
        ],
        "description": "Decline in mitochondrial function",
    },
    "cellular_senescence": {
        "keywords": ["senescence", "SASP", "p16", "p21", "senescent cells"],
        "description": "Accumulation of non-dividing cells",
    },
    "stem_cell_exhaustion": {
        "keywords": ["stem cell", "regeneration", "tissue repair", "progenitor"],
        "description": "Decline in stem cell function",
    },
    "altered_intercellular_communication": {
        "keywords": [
            "inflammation",
            "inflammaging",
            "cytokine",
            "immune",
            "neurohormonal",
        ],
        "description": "Changes in cell-cell signaling",
    },
    "chronic_inflammation": {
        "keywords": ["inflammaging", "NF-kB", "IL-6", "TNF", "chronic inflammation"],
        "description": "Persistent low-grade inflammation",
    },
    "dysbiosis": {
        "keywords": ["microbiome", "gut bacteria", "dysbiosis", "microbial"],
        "description": "Disruption of microbiome composition",
    },
}

# Top intervention targets based on current longevity research
LONGEVITY_TARGETS = {
    "mTOR_inhibition": ["rapamycin", "rapalogs", "mTOR inhibitor"],
    "NAD_boosters": ["NMN", "NR", "NAD+", "nicotinamide"],
    "senolytics": ["dasatinib", "quercetin", "fisetin", "senolytic"],
    "metformin": ["metformin", "biguanide"],
    "spermidine": ["spermidine", "polyamine"],
    "resveratrol": ["resveratrol", "SIRT1 activator"],
    "caloric_restriction": ["caloric restriction", "fasting", "dietary restriction"],
}
