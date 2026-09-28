#!/bin/zsh
# Run every track for every model in models.toml (models in parallel, tracks in sequence per model).
# Needs the API keys in the environment and the claude, codex and agy CLIs logged in.
# usage: ./run_all.sh [model ...]
cd "${0:A:h}"
mkdir -p logs
models=("$@")
[[ ${#models} -eq 0 ]] && models=($(grep -o '^\[models\.[^]]*' models.toml | cut -d. -f2-))
for m in $models; do
  (for t in mmlu bnexam belebele mgsm; do uv run python bench/run.py $t $m >> logs/$m.log 2>&1; done) &
done
wait
uv run python bench/score.py
