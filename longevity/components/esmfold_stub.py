# scripts/esmfold_stub.py
'''
ESMFold / AlphaFold integration stub.

Notes:
- Running full AlphaFold locally requires large databases and GPU memory; instead use:
   * ESMFold (esm package) for single-sequence structure prediction (lighter)
   * ColabFold (recommended) for accessible AlphaFold-like runs on Colab or cloud GPUs
- This script provides a wrapper to call ESMFold if installed, otherwise shows instructions.

Example usage (local ESMFold, if installed):
  python scripts/esmfold_stub.py --fasta inputs/target.fasta --out outputs/target.pdb

If you don't have ESMFold locally, use ColabFold:
- Open ColabFold notebook: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
- Upload sequences or connect to your Google Drive
- Run with GPU runtime.

'''
import argparse, os, sys
parser = argparse.ArgumentParser()
parser.add_argument("--fasta", required=False)
parser.add_argument("--out", required=False)
args = parser.parse_args()

try:
    import esm
    print("ESM package present. You may call esm.pretrained.esmfold_v1() for predictions. See ESM docs.")
except Exception as e:
    print("ESM/ESMFold not installed locally. See script header for ColabFold instructions.")
