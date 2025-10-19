# scripts/preprocess.py
import argparse, json
import spacy
import scispacy
from tqdm import tqdm
import re

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="inp", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

nlp = spacy.load("en_core_sci_sm")

def extract_text(pubrec):
    # pubrec is the Entrez XML parsed dict
    try:
        art = pubrec[0]["MedlineCitation"]["Article"]
        title = art.get("ArticleTitle","")
        abstract = ""
        if "Abstract" in art and art["Abstract"]:
            parts = art["Abstract"].get('AbstractText', [])
            if isinstance(parts, list):
                abstract = ' '.join(parts)
            else:
                abstract = str(parts)
        return (title, abstract)
    except Exception as e:
        return ("","")

out = []
with open(args.inp, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        title, abstract = extract_text(rec)
        text = (title or "") + "\n" + (abstract or "")
        # basic cleaning
        text = re.sub(r"\s+", " ", text).strip()
        doc = nlp(text)
        ents = [{"text":ent.text, "label": ent.label_} for ent in doc.ents]
        outrec = {"title": title, "abstract": abstract, "text": text, "entities": ents}
        out.append(outrec)
with open(args.out, "w", encoding='utf-8') as f:
    for r in out:
        f.write(json.dumps(r)+"\n")
print("Processed", len(out), "documents")
