"""Download the source datasets and build the item files for every track.

Question text is never committed: data/raw/ and data/items/ are gitignored, and this script
rebuilds them from the original publishers. Only the sampled ids (data/ids/) are tracked.

Tracks
  mmlu      OpenAI MMMLU Bengali (human translation of the MMLU test set) paired with English MMLU
  bnexam    hishab/bangla-mmlu: native Bangla questions from Bangladeshi exams (BCS, job, HSC, admission)
  belebele  Belebele reading comprehension, ben_Beng paired with eng_Latn on (link, question_number)
  mgsm      MGSM grade-school maths, Bengali paired with English by row
"""
import json
import random
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, ITEMS, IDS = ROOT / "data/raw", ROOT / "data/items", ROOT / "data/ids"
SEED = 20261006
SOURCES = {
    "mmlu_EN.csv": "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
    "mmlu_BN-BD.csv": "https://huggingface.co/datasets/openai/MMMLU/resolve/main/test/mmlu_BN-BD.csv",
    "bnmmlu_test.parquet": "https://huggingface.co/datasets/hishab/bangla-mmlu/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    "belebele_ben_Beng.jsonl": "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/ben_Beng.jsonl",
    "belebele_eng_Latn.jsonl": "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/eng_Latn.jsonl",
    "mgsm_bn.parquet": "https://huggingface.co/datasets/juletxara/mgsm/resolve/main/bn/test-00000-of-00001.parquet",
    "mgsm_en.parquet": "https://huggingface.co/datasets/juletxara/mgsm/resolve/main/en/test-00000-of-00001.parquet",
}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (RAW / name).exists():
            print("download", name)
            urllib.request.urlretrieve(url, RAW / name)


def mcq(qid, question, options, passage=None):
    head = f"[id {qid}]\n" + (f"Passage: {passage}\nQuestion: {question}\n" if passage else f"{question}\n")
    return head + "".join(f"{k}) {v}\n" for k, v in zip("ABCD", options))


def write(task, items):
    ITEMS.mkdir(parents=True, exist_ok=True)
    IDS.mkdir(parents=True, exist_ok=True)
    with open(ITEMS / f"{task}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    (IDS / f"{task}.json").write_text(json.dumps([it["id"] for it in items]))
    print(f"{task}: {len(items)} items, langs {sorted(items[0]['langs'])}")


def sampled(task, population, k, rng):
    """Reuse the committed id list if present, so every run scores the same items."""
    path = IDS / f"{task}.json"
    if path.exists():
        return json.loads(path.read_text())
    return sorted(rng.sample(population, k)) if k < len(population) else list(population)


def build_mmlu():
    en = pd.read_csv(RAW / "mmlu_EN.csv")
    bn = pd.read_csv(RAW / "mmlu_BN-BD.csv")
    assert len(en) == len(bn) and (en.Answer == bn.Answer).all() and (en.Subject == bn.Subject).all()
    path = IDS / "mmlu.json"
    if path.exists():
        ids = json.loads(path.read_text())
    else:  # 5 per subject, 57 subjects
        rng = random.Random(SEED)
        ids = []
        for subject in sorted(en.Subject.unique()):
            ids += sorted(rng.sample(en.index[en.Subject == subject].tolist(), 5))
    write("mmlu", [{
        "id": str(i), "type": "mcq", "answer": en.Answer[i], "meta": {"subject": en.Subject[i]},
        "langs": {lang: mcq(i, df.Question[i], [df[c][i] for c in "ABCD"]) for lang, df in (("en", en), ("bn", bn))},
    } for i in map(int, ids)])


def build_bnexam():
    df = pd.read_parquet(RAW / "bnmmlu_test.parquet")
    df = df[df.choices.map(len) == 4].reset_index(drop=True)
    by_id = {r.id: r for r in df.itertuples()}
    ids = sampled("bnexam", sorted(by_id), 300, random.Random(SEED))
    write("bnexam", [{
        "id": i, "type": "mcq", "answer": by_id[i].answer,
        "langs": {"bn": mcq(i, by_id[i].question, list(by_id[i].choices))},
    } for i in ids])


def build_belebele():
    rows = {}
    for lang, code in (("bn", "ben_Beng"), ("en", "eng_Latn")):
        with open(RAW / f"belebele_{code}.jsonl", encoding="utf-8") as f:
            for r in map(json.loads, f):
                rows.setdefault((r["link"], r["question_number"]), {})[lang] = r
    paired = {f"{n}:{link}": v for (link, n), v in rows.items() if len(v) == 2}
    assert all(v["bn"]["correct_answer_num"] == v["en"]["correct_answer_num"] for v in paired.values())
    ids = sampled("belebele", sorted(paired), 300, random.Random(SEED))
    items = []
    for n, i in enumerate(ids):
        v = paired[i]
        items.append({
            "id": str(n), "source_key": i, "type": "mcq", "answer": "ABCD"[int(v["en"]["correct_answer_num"]) - 1],
            "langs": {lang: mcq(n, r["question"], [r[f"mc_answer{k}"] for k in range(1, 5)], r["flores_passage"])
                      for lang, r in v.items()},
        })
    write("belebele", items)
    (IDS / "belebele.json").write_text(json.dumps(ids))


def build_mgsm():
    en = pd.read_parquet(RAW / "mgsm_en.parquet")
    bn = pd.read_parquet(RAW / "mgsm_bn.parquet")
    assert len(en) == len(bn) == 250 and (en.answer_number == bn.answer_number).all()
    write("mgsm", [{
        "id": str(i), "type": "num", "answer": float(en.answer_number[i]),
        "langs": {"en": f"[id {i}]\n{en.question[i]}\n", "bn": f"[id {i}]\n{bn.question[i]}\n"},
    } for i in range(250)])


if __name__ == "__main__":
    fetch()
    build_mmlu()
    build_bnexam()
    build_belebele()
    build_mgsm()
