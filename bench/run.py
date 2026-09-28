"""Run the paired English/Bengali MMLU sample through one model.

usage: uv run bench/run.py <model-key> [--lang en,bn] [--workers 4]

Models are defined in models.toml. API keys come from the environment only.
Each batch's raw reply is saved under results/raw/<model-key>/ so runs resume.
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
BATCH = 15
SYSTEM = "You are taking a multiple-choice exam. Answer every question."
INSTRUCTION = (
    "Answer the following {n} multiple-choice questions. Each has exactly one correct option.\n"
    "You may reason briefly, but the last line of your reply must be JSON only, of the form\n"
    '{{"answers": {{"<id>": "<A|B|C|D>", ...}}}} with one entry per question id.\n\n'
)

API_BASES = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "zai": ("https://api.z.ai/api/paas/v4", "ZAI_API_KEY"),
}


def load_items():
    with open(ROOT / "data/items.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_prompt(batch, lang):
    parts = [INSTRUCTION.format(n=len(batch))]
    for it in batch:
        q = it[lang]
        parts.append(
            f"[id {it['id']}]\n{q['Question']}\nA) {q['A']}\nB) {q['B']}\nC) {q['C']}\nD) {q['D']}\n"
        )
    return "\n".join(parts)


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
        headers={"Authorization": f"Bearer {os.environ[key_env]}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    return {"text": msg.get("content") or "", "usage": d.get("usage"), "served_by": d.get("provider")}


def call_cli(cli, model, prompt, extra):
    with tempfile.TemporaryDirectory() as tmp:
        if cli == "claude":
            cmd = ["claude", "-p", "--model", model, "--system-prompt", SYSTEM, "--tools", "",
                   "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                   "--output-format", "json", prompt]
            out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=1800,
                                 stdin=subprocess.DEVNULL)
            d = json.loads(out.stdout)
            return {"text": d.get("result") or "", "usage": d.get("usage"), "cost_usd": d.get("total_cost_usd")}
        if cli == "codex":
            last = Path(tmp) / "last.txt"
            cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
                   "--ignore-rules", "-m", model, "-c", f"model_reasoning_effort={extra.get('effort', 'medium')}",
                   "-s", "read-only", "-o", str(last), "--json", SYSTEM + "\n\n" + prompt]
            out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=1800,
                                 stdin=subprocess.DEVNULL)
            usage = [json.loads(l)["usage"] for l in out.stdout.splitlines() if '"usage"' in l]
            return {"text": last.read_text() if last.exists() else "", "usage": usage[-1] if usage else None,
                    "stderr_tail": out.stderr[-300:] if out.returncode else ""}
        if cli == "agy":
            cmd = ["agy", "--model", model, "--effort", extra.get("effort", "medium"), "--print-timeout", "20m",
                   "--output-format", "json", "-p", SYSTEM + "\n\n" + prompt]
            out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=1800,
                                 stdin=subprocess.DEVNULL)
            d = json.loads(out.stdout)
            return {"text": d.get("response") or "", "usage": d.get("usage")}
    raise ValueError(cli)


def parse_answers(text):
    for m in reversed(list(re.finditer(r"\{[^{}]*\"answers\"\s*:\s*\{[^{}]*\}\s*\}", text, re.S))):
        try:
            ans = json.loads(m.group(0))["answers"]
            return {str(k): str(v).strip().upper()[:1] for k, v in ans.items()}
        except json.JSONDecodeError:
            continue
    return {}


def run_batch(cfg, key, lang, idx, batch):
    path = ROOT / "results/raw" / key / f"{lang}_{idx:02d}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        if len(rec["answers"]) == len(batch):
            return rec
    prompt = build_prompt(batch, lang)
    t0 = time.time()
    r = {}
    for attempt in range(3):
        try:
            if "cli" in cfg:
                r = call_cli(cfg["cli"], cfg["model"], prompt, cfg)
            else:
                r = call_api(cfg["provider"], cfg["model"], prompt, cfg)
            break
        except Exception as e:  # network, rate limit, CLI hiccup: retry, then record the failure
            r = {"text": "", "error": repr(e)[:500]}
            time.sleep(10 * (attempt + 1))
    rec = {"model": key, "lang": lang, "batch": idx, "ids": [it["id"] for it in batch],
           "seconds": round(time.time() - t0, 1), **r, "answers": parse_answers(r.get("text", ""))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lang", default="en,bn")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N batches per language (pilot)")
    a = ap.parse_args()
    cfg = tomllib.loads((ROOT / "models.toml").read_text())["models"][a.model]
    items = load_items()
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    if a.limit:
        batches = batches[:a.limit]
    jobs = [(lang, i, b) for lang in a.lang.split(",") for i, b in enumerate(batches)]
    workers = cfg.get("workers", a.workers)
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(lambda j: run_batch(cfg, a.model, *j), jobs))
    for lang in a.lang.split(","):
        rs = [r for r in recs if r["lang"] == lang]
        n = sum(len(r["ids"]) for r in rs)
        got = sum(len(r["answers"]) for r in rs)
        errs = sum(1 for r in rs if r.get("error"))
        print(f"{a.model} {lang}: {got}/{n} answers parsed, {errs} failed batches")


if __name__ == "__main__":
    main()
