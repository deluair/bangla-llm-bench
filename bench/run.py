"""Run one track through one model.

usage: uv run bench/run.py <task> <model-key> [--lang en,bn] [--workers 4] [--limit N]

Tracks are built by bench/prepare.py; models are defined in models.toml; API keys come from the
environment only. Each batch's raw reply is saved under results/raw/<task>/<model-key>/, so runs resume.
Protocol: questions go in batches with a JSON answer line; any question the batch reply leaves
unanswered is re-asked alone (kept under "single" in the record) before it is scored as wrong.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH = {"mmlu": 15, "bnexam": 15, "belebele": 8, "mgsm": 10}
SYSTEM = "You are taking a multiple-choice exam. Answer every question."
INSTRUCTION = {
    "mcq": (
        "Answer the following {n} multiple-choice questions. Each has exactly one correct option.\n"
        "You may reason briefly, but the last line of your reply must be JSON only, of the form\n"
        '{{"answers": {{"<id>": "<A|B|C|D>", ...}}}} with one entry per question id.\n\n'
    ),
    "num": (
        "Solve the following {n} math word problems. Each answer is a single number.\n"
        "You may reason briefly, but the last line of your reply must be JSON only, of the form\n"
        '{{"answers": {{"<id>": <number>, ...}}}} with one entry per problem id, digits only.\n\n'
    ),
}
SINGLE = {
    "mcq": "Answer this multiple-choice question. Reply with only the letter of the correct option.\n\n",
    "num": "Solve this math word problem. End your reply with a line of the form: ANSWER: <number>\n\n",
}
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
NUM = r"-?[\d০-৯][\d০-৯,]*(?:\.[\d০-৯]+)?"

API_BASES = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "zai": ("https://api.z.ai/api/coding/paas/v4", "ZAI_API_KEY"),
}


def load_items(task):
    with open(ROOT / f"data/items/{task}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_prompt(batch, lang):
    return INSTRUCTION[batch[0]["type"]].format(n=len(batch)) + "\n".join(it["langs"][lang] for it in batch)


def call_api(provider, model, prompt, extra):
    base, key_env = API_BASES[provider]
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": extra.get("max_tokens", 16000),
    }
    body.update(extra.get("body", {}))
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ[key_env]}", "Content-Type": "application/json",
                 "User-Agent": "bangla-llm-bench/0.1"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    return {"text": msg.get("content") or "", "usage": d.get("usage"), "served_by": d.get("provider")}


# The CLIs run on their subscription logins; a stray vendor API key in the environment would override that.
CLI_ENV = {k: v for k, v in os.environ.items()
           if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"}}


def call_cli(cli, model, prompt, extra):
    with tempfile.TemporaryDirectory() as tmp:
        if cli == "claude":
            cmd = ["claude", "-p", "--model", model, "--system-prompt", SYSTEM, "--tools", "",
                   "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                   "--output-format", "json", prompt]
            out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=1800,
                                 stdin=subprocess.DEVNULL, check=False, env=CLI_ENV)
            d = json.loads(out.stdout)
            if d.get("is_error") or str(d.get("result", "")).startswith(("Failed to authenticate", "API Error")):
                raise RuntimeError(f"claude: {str(d.get('result'))[:300]}")
            return {"text": d.get("result") or "", "usage": d.get("usage"), "cost_usd": d.get("total_cost_usd")}
        if cli == "codex":
            last = Path(tmp) / "last.txt"
            cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
                   "--ignore-rules", "-m", model, "-c", f"model_reasoning_effort={extra.get('effort', 'medium')}",
                   "-s", "read-only", "-o", str(last), "--json", SYSTEM + "\n\n" + prompt]
            out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=1800,
                                 stdin=subprocess.DEVNULL, check=False, env=CLI_ENV)
            usage = [json.loads(line)["usage"] for line in out.stdout.splitlines() if '"usage"' in line]
            text = last.read_text() if last.exists() else ""
            if not text:
                errs = [line for line in out.stdout.splitlines() if '"error"' in line or "turn.failed" in line]
                raise RuntimeError(f"codex rc={out.returncode}: {(errs[-1] if errs else out.stderr)[-300:]}")
            return {"text": text, "usage": usage[-1] if usage else None}
        if cli == "agy":
            cmd = ["agy", "--model", model, "--effort", extra.get("effort", "medium"), "--print-timeout", "20m",
                   "--output-format", "json", "-p", SYSTEM + "\n\n" + prompt]
            out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=1800,
                                 stdin=subprocess.DEVNULL, check=False, env=CLI_ENV)
            d = json.loads(out.stdout)
            if not d.get("response"):
                raise RuntimeError(f"agy: {out.stderr[-300:]}")
            return {"text": d["response"], "usage": d.get("usage")}
    raise ValueError(cli)


def to_number(s):
    try:
        return float(s.translate(BN_DIGITS).replace(",", "").strip())
    except ValueError:
        return None


def parse_answers(text, kind="mcq"):
    # Lenient: take "id": answer pairs after the last "answers" key, so a missing brace is not a zero.
    at = text.rfind('"answers"')
    if at < 0:
        return {}
    key = r'"(?:\[?id\s*)?([\w-]+)\]?"\s*:\s*'
    if kind == "mcq":
        return {k: v.upper() for k, v in re.findall(key + r'"\s*\(?([A-Da-d])\b', text[at:])}
    pairs = re.findall(key + r'"?\s*\$?(' + NUM + ")", text[at:])
    return {k: n for k, v in pairs if (n := to_number(v)) is not None}


def parse_single(text, kind="mcq"):
    if kind == "mcq":
        letters = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text)
        return letters[-1] if letters else None
    m = re.findall(r"ANSWER:\s*\$?(" + NUM + ")", text)
    return to_number(m[-1]) if m else None


def call(cfg, prompt):
    r = {}
    for attempt in range(3):
        try:
            if "cli" in cfg:
                return call_cli(cfg["cli"], cfg["model"], prompt, cfg)
            return call_api(cfg["provider"], cfg["model"], prompt, cfg)
        except Exception as e:  # noqa: BLE001 -- network, rate limit, CLI hiccup: retry, then record the failure
            r = {"text": "", "error": repr(e)[:500]}
            time.sleep(15 * (attempt + 1))
    return r


def run_batch(cfg, task, key, lang, idx, batch):
    kind = batch[0]["type"]
    path = ROOT / "results/raw" / task / key / f"{lang}_{idx:02d}.json"
    rec = json.loads(path.read_text()) if path.exists() else None
    if rec is None or rec.get("error") or not rec.get("text"):
        t0 = time.time()
        r = call(cfg, build_prompt(batch, lang))
        rec = {"task": task, "model": key, "lang": lang, "batch": idx, "ids": [it["id"] for it in batch],
               "seconds": round(time.time() - t0, 1), **r}
    rec["answers_batch"] = parse_answers(rec.get("text", ""), kind)
    single = rec.get("single", {})
    for it in batch:
        k = it["id"]
        if k in rec["answers_batch"] or single.get(k, {}).get("answer") is not None:
            continue
        r = call(cfg, SINGLE[kind] + it["langs"][lang])
        single[k] = {"text": r.get("text", "")[-2000:], "error": r.get("error"), "usage": r.get("usage"),
                     "answer": parse_single(r.get("text", ""), kind)}
    rec["single"] = single
    rec["answers"] = {**{k: v["answer"] for k, v in single.items() if v.get("answer") is not None},
                      **rec["answers_batch"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(BATCH))
    ap.add_argument("model")
    ap.add_argument("--lang", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N batches per language (pilot)")
    a = ap.parse_args()
    cfg = tomllib.loads((ROOT / "models.toml").read_text())["models"][a.model]
    items = load_items(a.task)
    langs = a.lang.split(",") if a.lang else sorted(items[0]["langs"], reverse=True)
    size = BATCH[a.task]
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    if a.limit:
        batches = batches[:a.limit]
    jobs = [(lang, i, b) for lang in langs for i, b in enumerate(batches)]
    with ThreadPoolExecutor(cfg.get("workers", a.workers)) as ex:
        recs = list(ex.map(lambda j: run_batch(cfg, a.task, a.model, *j), jobs))
    for lang in langs:
        rs = [r for r in recs if r["lang"] == lang]
        n = sum(len(r["ids"]) for r in rs)
        print(f"{a.task} {a.model} {lang}: {sum(len(r['answers']) for r in rs)}/{n} answered, "
              f"{sum(len(r['single']) for r in rs)} re-asked singly, {sum(1 for r in rs if r.get('error'))} failed batches")


if __name__ == "__main__":
    main()
