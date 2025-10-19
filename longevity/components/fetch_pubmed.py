# scripts/fetch_pubmed.py
import argparse
import json
from time import sleep
import os

from Bio import Entrez

parser = argparse.ArgumentParser()
parser.add_argument("--query", required=True)
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--out", required=True)
args = parser.parse_args()

email = None
email = os.getenv("ENTRZ_EMAIL") or os.getenv("ENTREZ_EMAIL")
if not email:
    raise SystemExit(
        "Set ENT RZ_EMAIL or ENTREZ_EMAIL environment variable to your email (NCBI requirement)."
    )
Entrez.email = email


def fetch_ids(q, n):
    handle = Entrez.esearch(db="pubmed", term=q, retmax=n)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_abstract(pid):
    handle = Entrez.efetch(db="pubmed", id=pid, rettype="abstract", retmode="xml")
    rec = Entrez.read(handle)
    handle.close()
    return rec


ids = fetch_ids(args.query, args.n)
out = []
for i, pid in enumerate(ids):
    try:
        rec = fetch_abstract(pid)
        out.append(rec)
    except Exception as e:
        print("failed id", pid, e)
    sleep(0.34)  # rate-limit
with open(args.out, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r))
        f.write("\n")
print("Wrote", len(out), "records to", args.out)
