"""Token premium: how many more tokens the same MMLU items take in Bengali than in English.

Counts question plus four options for all sampled items with each published tokenizer.
Writes results/tokens.csv.
"""
import json
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "data/items/mmlu.jsonl", encoding="utf-8") as fh:
    items = [json.loads(line) for line in fh]


def text(it, lang):
    # Prompt block is "[id N]\n<question>\nA) ..\nB) ..\nC) ..\nD) ..\n"; count question and options only.
    lines = it["langs"][lang].rstrip("\n").split("\n")[1:]
    return "\n".join(["\n".join(lines[:-4])] + [o[3:] for o in lines[-4:]])


# (label, loader). HF repos are the vendors' own where ungated, else an exact public mirror.
TOKENIZERS = [
    ("OpenAI o200k (GPT-4o and later)", lambda: tiktoken.get_encoding("o200k_base").encode),
    ("OpenAI cl100k (GPT-4, GPT-3.5)", lambda: tiktoken.get_encoding("cl100k_base").encode),
    ("Qwen3", lambda: Tokenizer.from_pretrained("Qwen/Qwen3-8B").encode),
    ("DeepSeek-V3", lambda: Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V3").encode),
    ("Gemma 3", lambda: Tokenizer.from_pretrained("unsloth/gemma-3-4b-it").encode),
    ("Llama 3.1", lambda: Tokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct").encode),
    ("Mistral Nemo (Tekken)", lambda: Tokenizer.from_pretrained("unsloth/Mistral-Nemo-Instruct-2407").encode),
]

rows = []
for label, load in TOKENIZERS:
    enc = load()
    n = {lang: sum(len(enc(text(it, lang)) if label.startswith("OpenAI") else enc(text(it, lang)).ids)
                   for it in items) for lang in ("en", "bn")}
    rows.append((label, n["en"], n["bn"], round(n["bn"] / n["en"], 2)))
    print(f"{label:34s} en {n['en']:7d}  bn {n['bn']:7d}  ratio {n['bn'] / n['en']:.2f}")

chars = {lang: sum(len(text(it, lang)) for it in items) for lang in ("en", "bn")}
print(f"characters: en {chars['en']}, bn {chars['bn']}")
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results/tokens.csv", "w") as f:
    f.write("tokenizer,tokens_en,tokens_bn,bn_over_en\n")
    f.writelines(",".join(map(str, r)) + "\n" for r in rows)
