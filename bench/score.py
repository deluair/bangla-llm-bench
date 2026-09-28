"""Score every model on every track; write results/scores.csv and results/leaderboard.md."""
import json
import random
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKS = {"mmlu": "MMLU", "belebele": "Belebele", "mgsm": "MGSM", "bnexam": "BD exams"}
PAIRED = ["mmlu", "belebele", "mgsm"]
roster = tomllib.loads((ROOT / "models.toml").read_text())["models"]


def load_items(task):
    with open(ROOT / f"data/items/{task}.jsonl", encoding="utf-8") as f:
        return {it["id"]: it for it in map(json.loads, f)}


def correct(it, pred):
    if pred is None:
        return 0
    if it["type"] == "num":
        return int(abs(float(pred) - it["answer"]) < 1e-6)
    return int(pred == it["answer"])


BATCH = {"mmlu": 15, "bnexam": 15, "belebele": 8, "mgsm": 10}


def finished(r):
    """A batch is final once its single re-asks ran; an errored batch only if every leftover failure is an
    HTTP 400 (the provider refusing the request, e.g. a content filter), which no retry will change."""
    if "single" not in r:
        return False
    return not r.get("error") or all(v.get("answer") is not None or "HTTPError 400" in (v.get("error") or "")
                                     for v in r["single"].values())


def load_run(task, key):
    """Answers per language, or None for a language whose batches are not all finished."""
    got, ptok, cost, single, done = {}, {}, 0.0, 0, {}
    for f in sorted((ROOT / "results/raw" / task / key).glob("*.json")):
        r = json.loads(f.read_text())
        if finished(r):
            done[r["lang"]] = done.get(r["lang"], 0) + 1
        got.setdefault(r["lang"], {}).update(r["answers"])
        single += sum(1 for k, v in r.get("single", {}).items()
                      if v.get("answer") is not None and k not in r.get("answers_batch", {}))
        u = r.get("usage") or {}
        ptok[r["lang"]] = ptok.get(r["lang"], 0) + (u.get("prompt_tokens") or 0)
        cost += u.get("cost") or r.get("cost_usd") or 0
    return got, ptok, cost, single, done


def bootstrap(diffs, reps=5000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


rows = []
for task in TRACKS:
    items = load_items(task)
    n = len(items)
    for key, cfg in roster.items():
        if not (ROOT / "results/raw" / task / key).exists():
            continue
        got, ptok, cost, single, done = load_run(task, key)
        langs = sorted(next(iter(items.values()))["langs"])
        need = -(-n // BATCH[task])
        if any(done.get(lang, 0) < need for lang in langs):
            print(f"incomplete: {task} {key} " + " ".join(f"{lang}={done.get(lang, 0)}/{need} batches" for lang in langs))
            continue
        ok = {lang: {i: correct(it, got[lang].get(i)) for i, it in items.items()} for lang in langs}
        row = {"task": task, "model": key, "size": cfg["size"], "route": cfg.get("cli") or cfg["provider"], "n": n,
               "bn": sum(ok["bn"].values()) / n, "re_asked": single,
               "unanswered": sum(n - len(got[lang]) for lang in langs), "api_cost_usd": round(cost, 4)}
        if "en" in ok:
            row["en"] = sum(ok["en"].values()) / n
            row["gap"] = row["bn"] - row["en"]
            row["gap_lo"], row["gap_hi"] = bootstrap([ok["bn"][i] - ok["en"][i] for i in items])
            if ptok.get("en"):
                row["bn_en_prompt_tokens"] = ptok["bn"] / ptok["en"]
        rows.append(row)

(ROOT / "results").mkdir(exist_ok=True)
cols = ["task", "model", "size", "route", "n", "en", "bn", "gap", "gap_lo", "gap_hi", "re_asked", "unanswered",
        "bn_en_prompt_tokens", "api_cost_usd"]
with open(ROOT / "results/scores.csv", "w") as f:
    f.write(",".join(cols) + "\n")
    f.writelines(",".join("" if r.get(c) is None else (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))
                          for c in cols) + "\n" for r in rows)

by = {}
for r in rows:
    by.setdefault(r["model"], {})[r["task"]] = r
complete = {m: t for m, t in by.items() if all(k in t for k in TRACKS)}


def pct(x):
    return f"{100 * x:.1f}"


lines = ["| Rank | Model | Size | Bangla score | " + " | ".join(
    f"{TRACKS[t]} en | {TRACKS[t]} bn | gap" for t in PAIRED) + " | BD exams bn | Mean gap |",
         "|" + "---|" * (4 + 3 * len(PAIRED) + 2)]
order = sorted(complete, key=lambda m: -sum(complete[m][t]["bn"] for t in TRACKS))
for rank, m in enumerate(order, 1):
    t = complete[m]
    score = sum(t[k]["bn"] for k in TRACKS) / len(TRACKS)
    gaps = sum(t[k]["gap"] for k in PAIRED) / len(PAIRED)
    cells = [f"{pct(t[k]['en'])} | {pct(t[k]['bn'])} | {100 * t[k]['gap']:+.1f}" for k in PAIRED]
    lines.append(f"| {rank} | {m} | {t['mmlu']['size']} | **{pct(score)}** | " + " | ".join(cells)
                 + f" | {pct(t['bnexam']['bn'])} | {100 * gaps:+.1f} |")
(ROOT / "results/leaderboard.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n{len(complete)} models complete on all tracks; {len(by) - len(complete)} partial")
