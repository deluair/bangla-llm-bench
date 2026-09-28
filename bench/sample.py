"""Draw a fixed stratified sample of MMLU items, paired English/Bengali."""
import json
import random
from pathlib import Path

import pandas as pd

PER_SUBJECT = 5
SEED = 20261006

raw = Path("data/raw")
en = pd.read_csv(raw / "mmlu_EN.csv")
bn = pd.read_csv(raw / "mmlu_BN-BD.csv")
assert len(en) == len(bn) and (en.Answer == bn.Answer).all() and (en.Subject == bn.Subject).all()

rng = random.Random(SEED)
ids = []
for subject in sorted(en.Subject.unique()):
    rows = en.index[en.Subject == subject].tolist()
    ids += sorted(rng.sample(rows, PER_SUBJECT))

with open("data/items.jsonl", "w", encoding="utf-8") as f:
    for i in ids:
        f.write(json.dumps({
            "id": int(i),
            "subject": en.Subject[i],
            "answer": en.Answer[i],
            "en": {k: str(en[k][i]) for k in ["Question", "A", "B", "C", "D"]},
            "bn": {k: str(bn[k][i]) for k in ["Question", "A", "B", "C", "D"]},
        }, ensure_ascii=False) + "\n")
print(f"{len(ids)} items from {en.Subject.nunique()} subjects -> data/items.jsonl")
