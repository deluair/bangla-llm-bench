# bangla-llm-bench

**Same questions, two languages.** How much worse do today's language models do in Bengali (Bangla) than in English, and how much more does each Bengali question cost?

The suite scores current frontier, mid-size and laptop-size models on four tracks. Three tracks pair every Bengali item with its English original, so the Bengali penalty is measured question by question. The fourth track uses native Bangla questions from Bangladeshi exams. It is scored in Bengali only.

<!-- LEADERBOARD -->
## Leaderboard

Results as of 2026-09-28. Numbers are percentages; gaps are Bengali minus English in points. Per-track 95% intervals are in `results/scores.csv`. Still running: `claude-haiku-4-5`, `kimi-k2-6`, `qwen3-5-9b`. `gpt-6-astra` ran out of its ChatGPT-plan quota mid-run and is rerun after 2026-10-03.

| Rank | Model | Size | Bangla score | MMLU en | MMLU bn | gap | Belebele en | Belebele bn | gap | MGSM en | MGSM bn | gap | BD exams bn | Mean gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gemini-3-8-flash | frontier | **96.1** | 95.1 | 94.7 | -0.4 | 98.7 | 97.3 | -1.3 | 100.0 | 99.6 | -0.4 | 92.7 | -0.7 |
| 2 | claude-opus-5-5 | frontier | **94.4** | 91.9 | 90.5 | -1.4 | 98.0 | 96.3 | -1.7 | 99.6 | 98.8 | -0.8 | 92.0 | -1.3 |
| 3 | gemini-3-1-pro | frontier | **93.8** | 93.7 | 91.2 | -2.5 | 98.0 | 94.7 | -3.3 | 99.6 | 96.8 | -2.8 | 92.7 | -2.9 |
| 4 | deepseek-v4-1-flash | mid | **91.8** | 89.8 | 87.0 | -2.8 | 97.3 | 93.7 | -3.7 | 98.4 | 97.6 | -0.8 | 89.0 | -2.4 |
| 5 | glm-5-3 | mid | **89.8** | 89.8 | 85.6 | -4.2 | 96.7 | 92.3 | -4.3 | 99.6 | 94.8 | -4.8 | 86.3 | -4.4 |
| 6 | claude-sonnet-5 | frontier | **89.1** | 91.2 | 85.3 | -6.0 | 95.3 | 87.7 | -7.7 | 98.0 | 94.8 | -3.2 | 88.7 | -5.6 |
| 7 | qwen3-8-27b | mid | **84.7** | 84.9 | 80.4 | -4.6 | 96.3 | 90.3 | -6.0 | 92.4 | 92.0 | -0.4 | 76.0 | -3.7 |
| 8 | gpt-oss-120b | mid | **83.1** | 86.0 | 79.3 | -6.7 | 97.0 | 86.0 | -11.0 | 98.0 | 91.6 | -6.4 | 75.7 | -8.0 |
| 9 | gemma-4-26b-a4b | small | **82.2** | 85.6 | 75.4 | -10.2 | 98.0 | 87.3 | -10.7 | 97.6 | 89.6 | -8.0 | 76.3 | -9.6 |
| 10 | gpt-oss-20b | small | **79.3** | 81.4 | 73.0 | -8.4 | 95.3 | 84.3 | -11.0 | 94.8 | 91.6 | -3.2 | 68.3 | -7.5 |
| 11 | qwen3-8b | small | **74.3** | 80.4 | 66.7 | -13.7 | 95.3 | 78.3 | -17.0 | 93.6 | 85.2 | -8.4 | 67.0 | -13.0 |
| 12 | ministral-8b | small | **68.4** | 77.5 | 64.2 | -13.3 | 92.0 | 80.3 | -11.7 | 89.2 | 75.2 | -14.0 | 54.0 | -13.0 |
| 13 | gemma-3-4b | small | **51.6** | 55.8 | 43.9 | -11.9 | 81.7 | 59.3 | -22.3 | 87.6 | 60.8 | -26.8 | 42.3 | -20.4 |
| 14 | llama-3-1-8b | small | **38.2** | 61.4 | 39.3 | -22.1 | 81.0 | 50.7 | -30.3 | 57.2 | 29.2 | -28.0 | 33.7 | -26.8 |
<!-- /LEADERBOARD -->

## Tracks

| Track | What it measures | Items | Bengali source | Pairing |
|---|---|---|---|---|
| **MMLU** | Broad knowledge, 57 subjects | 285 (5 per subject) | [OpenAI MMMLU](https://huggingface.co/datasets/openai/MMMLU) `BN_BD`, professional human translation of the MMLU test set | Same row in [English MMLU](https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv); answers and subjects match on all 14,042 rows |
| **Belebele** | Reading comprehension of a passage | 300 of 900 | [Belebele](https://huggingface.co/datasets/facebook/belebele) `ben_Beng`, human translation of FLORES passages | Joined to `eng_Latn` on (link, question_number); answer keys agree on every pair |
| **MGSM** | Grade-school maths word problems, numeric answer | 250 (all) | [MGSM](https://huggingface.co/datasets/juletxara/mgsm) `bn`, human translation of GSM8K | Same row in `en`; answers agree on all 250 |
| **BD exams** | Native Bangla knowledge from Bangladeshi exams (BCS and job exams, HSC, university, medical and engineering admission) | 300 of 14,750 | [hishab/bangla-mmlu](https://huggingface.co/datasets/hishab/bangla-mmlu) test split ([TituLLMs, arXiv:2502.11187](https://arxiv.org/abs/2502.11187)) | Native, no English pair |

Samples are drawn once with a fixed seed (`20261006`). The sampled ids are committed in `data/ids/`, so every model sees the same items.

**Bangla score** is the mean Bengali accuracy over the four tracks. **Gap** is Bengali accuracy minus English accuracy on the same items, in percentage points. The 95% interval comes from a paired bootstrap over items (5,000 resamples).

## Token premium

The same content costs more tokens in Bengali. Counted with each vendor's published tokenizer on the 285 MMLU items (question plus four options):

| Tokenizer | English tokens | Bengali tokens | Bengali ÷ English |
|---|---:|---:|---:|
| Gemma 3 | 23,004 | 27,012 | 1.17 |
| OpenAI o200k (GPT-4o and later) | 22,060 | 35,679 | 1.62 |
| Mistral Nemo (Tekken) | 23,080 | 42,963 | 1.86 |
| DeepSeek-V3 | 21,951 | 41,614 | 1.90 |
| Qwen3 | 22,679 | 101,727 | 4.49 |
| Llama 3.1 | 22,469 | 115,158 | 5.13 |
| OpenAI cl100k (GPT-4, GPT-3.5) | 22,199 | 115,861 | 5.22 |

The Bengali text has fewer characters than the English (96,722 vs 101,342), so the premium comes from the tokenizers, not from Bengali being longer.

## Protocol

- **One prompt format for every model.** Questions go in batches (15 for MMLU and BD exams, 8 for Belebele, 10 for MGSM). The model must end its reply with a JSON line `{"answers": {"<id>": "<letter or number>"}}`. Instructions are in English for both languages, so only the question text changes, as in OpenAI's `simple-evals` MMMLU setup.
- **No model is scored on formatting.** A question left out of the batch reply is re-asked on its own ("reply with only the letter", or "end with ANSWER: <number>"). It counts as wrong only if that also fails. The number re-asked is reported per model. Bengali digits (০-৯) are accepted in numeric answers.
- **Defaults.** Each model runs at its provider's default settings, with reasoning where the model reasons by default. Effort levels set for CLI models are listed in `models.toml`.
- **Routes.** Claude, GPT and Gemini models run through their vendors' own CLIs (`claude`, `codex`, `agy`) on subscription logins. Claude runs with a one-line system prompt, no tools and no MCP servers. Codex and Antigravity add their own harness system prompts, which cannot be removed from the CLI. Open-weight and other models run through each provider's OpenAI-compatible API: DeepSeek, Z.ai, Moonshot and Groq directly, the rest through OpenRouter. OpenRouter's serving provider is recorded for each batch.
- **Everything is kept.** Every raw reply, token count and cost is in `results/raw/<track>/<model>/`.

## Caveats

- **Translation noise.** Even professional translation garbles some items. MMLU item 10 renders the coset `|Ha|` as `|হা|` (Bengali letters). The paired design shows the penalty a Bengali speaker actually meets, but a little of it is translation error, not model weakness.
- **Refusals count as wrong.** A request the provider rejects outright (HTTP 400) is final and scored wrong. Z.ai's content filter rejects Belebele item 87 for GLM-5.3 in both languages (error 1301, "potentially unsafe or sensitive content"), so it costs GLM-5.3 one item on each side of the pair.
- **Samples, not full sets.** With about 300 items per track, a single accuracy has a standard error of about 2 to 3 points. Read the confidence intervals before reading the ranks.
- **Contamination.** MMLU, GSM8K and FLORES are public and old, and English originals are likelier to be in training data than Bengali translations. That would widen the gap. The native BD-exam track has no English original, so it is free of this particular bias.
- **Not covered yet.** Free-form Bengali generation (writing, summarisation, translation quality) and safety. FLORES+ translation is gated, and judged generation needs a careful rubric. Both are on the roadmap below.

## Reproduce

```bash
uv run --with pandas --with pyarrow python bench/prepare.py   # downloads sources into data/raw, builds data/items
export OPENROUTER_API_KEY=... DEEPSEEK_API_KEY=... ZAI_API_KEY=... MOONSHOT_API_KEY=... GROQ_API_KEY=...
./run_all.sh                        # or: ./run_all.sh <model-key> ...
uv run python bench/score.py        # results/scores.csv, results/leaderboard.md
uv run --with tiktoken --with tokenizers python bench/tokens.py
```

Question text is not redistributed here. `prepare.py` downloads it from the original publishers.

## Roadmap

1. Translation track (English↔Bengali, chrF++) once FLORES+ access is granted.
2. Judged Bengali writing: spelling, grammar and register, scored against a rubric by native speakers on a sample.
3. Bangladesh-specific knowledge written from primary sources (constitution, economy, geography, history), each item cited.
4. Safety in Bangla ([BanglaSafe](https://huggingface.co/datasets/BanglaLLM/BanglaSafe), [BanglaSocialBias](https://huggingface.co/datasets/csebuetnlp/BanglaSocialBias)).

## Licences

Code: MIT. Source datasets keep their own licences: MMMLU (MIT), Belebele (CC BY-SA 4.0), MGSM (CC BY-SA 4.0). hishab/bangla-mmlu states no licence on its dataset card, which is one reason no question text is copied into this repository.

## Citation

If you use the results, cite this repository and the source datasets: Hendrycks et al. 2021 (MMLU), OpenAI MMMLU, Bandarkar et al. 2024 (Belebele), Shi et al. 2023 (MGSM), and Nahin et al. 2025 (TituLLMs, arXiv:2502.11187).
