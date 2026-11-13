# Run both RAG experiments on the university cluster

Run these sections in order. Commands marked **Mac** run in your local terminal;
the others run over SSH on the cluster. Stop at a failed command and retain its
log before continuing. This first trial uses the same five practice queries.

The two implementations currently live in these local folders:

```text
edin-mls-26-autumn-coursework/rag/rag-adaptive-context-budget/rag/
edin-mls-26-autumn-coursework/rag/rag-dynamic-corpus/rag/
```

Each contains its own RAG runtime, shared-file changes, extension and tests. Copy
them into separate experiment directories. This preserves the versions that were
tested independently and avoids overwriting one task's shared files with the
other task's versions. Both use the cluster checkout's existing assets and ASR.

## 1. Mac: copy both implementations

The course guide uses `mlp.inf.ed.ac.uk` as the cluster head node. These commands
reach it through the Informatics SSH gateway. If you already have a working SSH
alias, you can use that route instead, consistently in SSH and rsync.

```bash
RAG_LOCAL_REPO="/Users/burphh/Desktop/CS/MLS-CW/edin-mls-26-autumn-coursework"
RAG_REMOTE="s2427231@mlp.inf.ed.ac.uk"
RAG_REMOTE_ROOT="/home/s2427231/edin-mls-26-autumn"

ssh -J s2427231@ssh.inf.ed.ac.uk "$RAG_REMOTE" \
  "mkdir -p '$RAG_REMOTE_ROOT/experiments/rag_cluster/adaptive_context_budget/rag' '$RAG_REMOTE_ROOT/experiments/rag_cluster/dynamic_corpus/rag'"

rsync -avz --progress -e "ssh -J s2427231@ssh.inf.ed.ac.uk" \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/' \
  --exclude='.cache/' --exclude='.venv/' --exclude='.mplconfig/' \
  --exclude='.env' --exclude='.env.local' \
  --exclude='data/' --exclude='models/' --exclude='results/' \
  "$RAG_LOCAL_REPO/rag/rag-adaptive-context-budget/rag/" \
  "$RAG_REMOTE:$RAG_REMOTE_ROOT/experiments/rag_cluster/adaptive_context_budget/rag/"

rsync -avz --progress -e "ssh -J s2427231@ssh.inf.ed.ac.uk" \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/' \
  --exclude='.cache/' --exclude='.venv/' --exclude='.mplconfig/' \
  --exclude='.env' --exclude='.env.local' \
  --exclude='data/' --exclude='models/' --exclude='results/' \
  "$RAG_LOCAL_REPO/rag/rag-dynamic-corpus/rag/" \
  "$RAG_REMOTE:$RAG_REMOTE_ROOT/experiments/rag_cluster/dynamic_corpus/rag/"

ssh -J s2427231@ssh.inf.ed.ac.uk "$RAG_REMOTE"
```

These commands **copy** files and leave the Mac copies intact. They include code,
tests, documentation and the existing labelled measurement archives. Large course
assets, credentials, caches and generated `results/` directories are excluded.
There is no `--delete`, Git push or merge. The inner `rag/` trailing slash matters.

## 2. Cluster head node: initialise Python environments

```bash
bash
cd /home/s2427231/edin-mls-26-autumn

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

conda info --envs
conda activate mls-asr
```

If `mls-asr` does not exist, run the supplied one-time setup **on the head node**,
then activate it. This downloads the ASR dependencies/model:

```bash
bash hw1-asr/setup_env.sh
# If setup installed Miniconda, initialise it in this shell too:
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls-asr
```

Once `mls-asr` is active, create/reuse the separate RAG environment:

```bash
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"

if [ ! -f "$HOME/mls-rag-env/bin/activate" ]; then
  python -m venv "$HOME/mls-rag-env"
fi
source "$HOME/mls-rag-env/bin/activate"

python -m pip install \
  -r experiments/rag_cluster/adaptive_context_budget/rag/requirements.txt \
  -r experiments/rag_cluster/adaptive_context_budget/rag/extensions/adaptive_context_budget/requirements-experiments.txt \
  -r experiments/rag_cluster/dynamic_corpus/rag/extensions/dynamic_corpus/requirements.txt
```

## 3. Cluster head node: install/reuse course assets and configure ELM

Use the canonical checkout's `rag/data/` and `rag/models/`. If the corpus, audio
and enc_a payload are already installed there, skip the download/extraction and
proceed to ELM configuration. Otherwise:

```bash
cd /home/s2427231/edin-mls-26-autumn
git lfs install
git lfs pull --include="rag-assets.zip"
mkdir -p "$HOME/mls-rag-assets"
unzip -n rag-assets.zip -d "$HOME/mls-rag-assets"
python rag/scripts/install_assets.py --from-dir "$HOME/mls-rag-assets"
```

Configure the existing course endpoint/model and your own key once:

```bash
if [ ! -f rag/.env ]; then
  cp rag/.env.example rag/.env
fi
chmod 600 rag/.env
nano rag/.env
```

Set `ELM_API_KEY` to your actual key and retain the endpoint/model settings you
have been given. Do not paste the key into commands or logs. No ELM calls have
been made by the steps above.

## 4. Cluster head node: share assets and create a reusable activation file

```bash
cd /home/s2427231/edin-mls-26-autumn

for project in adaptive_context_budget dynamic_corpus; do
  for asset in data models .env; do
    destination="$PWD/experiments/rag_cluster/$project/rag/$asset"
    if [ ! -e "$destination" ] && [ ! -L "$destination" ]; then
      ln -s "$PWD/rag/$asset" "$destination"
    fi
  done
done

cat > experiments/rag_cluster/activate.sh <<'BASH'
if [ -n "${VIRTUAL_ENV:-}" ] && type deactivate >/dev/null 2>&1; then
  deactivate
fi
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  printf '%s\n' 'Conda initializer missing; initialise the installed conda first.' >&2
  return 1
fi
conda activate mls-asr || return 1
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source "$HOME/mls-rag-env/bin/activate" || return 1

export MLS_CW_ROOT=/home/s2427231/edin-mls-26-autumn
export MLS_BUDGET_ROOT="$MLS_CW_ROOT/experiments/rag_cluster/adaptive_context_budget"
export MLS_DYNAMIC_ROOT="$MLS_CW_ROOT/experiments/rag_cluster/dynamic_corpus"
export MLS_RAG_RUN_ID=practice5_20261112_01
export MLS_RAG_RUN_ROOT="$MLS_CW_ROOT/experiments/rag_cluster/runs/$MLS_RAG_RUN_ID"
export RAG_ASR_ROOT="$MLS_CW_ROOT/hw1-asr"
export ASR_CUDA=none
export PATH="/opt/cuda-12.8.0/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$MLS_RAG_RUN_ROOT/matplotlib_cache"
mkdir -p "$MLS_RAG_RUN_ROOT/logs" "$MLS_RAG_RUN_ROOT/inputs" "$MPLCONFIGDIR"
set -o pipefail
BASH

source experiments/rag_cluster/activate.sh

cd "$MLS_BUDGET_ROOT"
python rag/scripts/check_setup.py 2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/setup_budget.log"

cd "$MLS_DYNAMIC_ROOT"
python rag/scripts/check_setup.py 2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/setup_dynamic.log"
```

Both setup checks should pass; these checks do not request ELM answers. `ASR_CUDA=none`
selects the supplied **PyTorch ASR on the GPU**, independently of coursework CUDA
kernel completion. No CUDA build is required for this RAG control.

Keep this run ID for all sections. For a fresh trial, change its line in
`activate.sh` before starting. Runners reject existing output directories; preserve
failed/partial runs rather than deleting them and confusing their provenance.

## 5. GPU allocation: tests and five real transcripts

From the cluster head-node shell:

```bash
srun -p Teaching --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
source /home/s2427231/edin-mls-26-autumn/experiments/rag_cluster/activate.sh

nvidia-smi -L | tee "$MLS_RAG_RUN_ROOT/logs/gpu.txt"
lscpu > "$MLS_RAG_RUN_ROOT/logs/cpu.txt"
python --version > "$MLS_RAG_RUN_ROOT/logs/python_version.txt" 2>&1
python -m pip freeze > "$MLS_RAG_RUN_ROOT/logs/python_packages.txt"
"$RAG_ASR_PYTHON" -c 'import torch; print("ASR PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available())' \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/asr_environment.txt"

cd "$MLS_BUDGET_ROOT"
PYTHONPATH=rag python -m unittest discover -s rag/tests -v \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/test_budget.log"

cd "$MLS_DYNAMIC_ROOT"
PYTHONPATH="$MLS_DYNAMIC_ROOT/rag:$MLS_BUDGET_ROOT/rag" \
python -m unittest discover -s rag/tests -v \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/test_dynamic.log"

PYTHONPATH=rag python -m extensions.dynamic_corpus prepare-queries \
  --split practice --limit 5 --device cpu --transcribe-missing \
  --transcript-cache "$MLS_RAG_RUN_ROOT/inputs/practice5_transcripts.jsonl" \
  --output "$MLS_RAG_RUN_ROOT/inputs/practice5_queries.npz" \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/prepare_queries.log"

cd "$MLS_BUDGET_ROOT"
PYTHONPATH=rag python -m extensions.adaptive_context_budget prepare \
  --split practice --limit 5 --k 5 --device cpu \
  --transcript-cache "$MLS_RAG_RUN_ROOT/inputs/practice5_transcripts.jsonl" \
  --output "$MLS_RAG_RUN_ROOT/inputs/budget_practice5.jsonl" \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/prepare_budget.log"

cat "$MLS_RAG_RUN_ROOT/inputs/budget_practice5.lengths.csv"
```

This transcribes only five practice recordings, then reuses those real transcripts
for both investigations. It does not run the teammate's full retrieval K sweep.
The RAG query encoder stays on CPU, leaving the GPU to ASR. No ELM requests yet.
The context-length table shows whether proposed budgets actually reduce evidence.

## 6. Same allocation: non-ELM experiments on actual course data

Dynamic corpus: all 21 cases, three strategies, three repeats, fixed k=10, using
real enc_a corpus vectors and the prepared real query embeddings:

```bash
cd "$MLS_DYNAMIC_ROOT"
PYTHONPATH=rag python -m extensions.dynamic_corpus ablate \
  --dataset course --k 10 --repeats 3 \
  --queries "$MLS_RAG_RUN_ROOT/inputs/practice5_queries.npz" \
  --output "$MLS_RAG_RUN_ROOT/dynamic_course" --plot \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/dynamic_course.log"
```

Budget: fixed-budget/packing/adaptive/matched-cost comparisons over frozen real
inputs, plus a separately labelled synthetic controller-overhead measurement:

```bash
cd "$MLS_BUDGET_ROOT"
PYTHONPATH=rag python -m extensions.adaptive_context_budget run \
  --snapshot "$MLS_RAG_RUN_ROOT/inputs/budget_practice5.jsonl" \
  --budgets 256 512 1024 2048 --packing-policies skip sentence_prefix \
  --match-adaptive --qrels rag/data/rag_corpus/qrels_practice.tsv \
  --output "$MLS_RAG_RUN_ROOT/budget_offline" \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/budget_offline.log"

PYTHONPATH=rag python -m extensions.adaptive_context_budget run \
  --snapshot "$MLS_RAG_RUN_ROOT/inputs/budget_practice5.jsonl" \
  --min-budget 128 --max-budget 1024 --match-adaptive \
  --output "$MLS_RAG_RUN_ROOT/budget_offline_adaptive128_1024" \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/budget_adaptive128_1024.log"

PYTHONPATH=rag python -m extensions.adaptive_context_budget run \
  --snapshot "$MLS_RAG_RUN_ROOT/inputs/budget_practice5.jsonl" \
  --min-budget 512 --max-budget 2048 --match-adaptive \
  --output "$MLS_RAG_RUN_ROOT/budget_offline_adaptive512_2048" \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/budget_adaptive512_2048.log"

PYTHONPATH=rag python -m extensions.adaptive_context_budget overhead \
  --counts 1 5 20 100 500 1000 --repeats 30 \
  --output "$MLS_RAG_RUN_ROOT/budget_overhead" \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/budget_overhead.log"

PYTHONPATH=rag python -m extensions.adaptive_context_budget summarise \
  --run-dir "$MLS_RAG_RUN_ROOT/budget_offline"

PYTHONPATH=rag python -m extensions.adaptive_context_budget plot \
  --run-dir "$MLS_RAG_RUN_ROOT/budget_offline" \
  --overhead-csv "$MLS_RAG_RUN_ROOT/budget_overhead/overhead.csv" \
  --output "$MLS_RAG_RUN_ROOT/budget_offline/figures"
```

These commands make **zero ELM calls**. Budget plots correctly omit answer-quality
and generation-latency axes until real generation/judge data exists. The corpus
retrieval/maintenance benchmarks are CPU computations even within a GPU allocation.

Return to the head node after these commands complete:

```bash
exit
```

## 7. Cluster head node: actual ELM generation

The repository guide notes that GPU nodes may lack outbound internet access.
Generation and judging below reuse the cached real inputs, so they do not need a
GPU; run them on the head node with API connectivity. This is a five-query pilot,
not a full-split API sweep.

```bash
source /home/s2427231/edin-mls-26-autumn/experiments/rag_cluster/activate.sh
cd "$MLS_BUDGET_ROOT"

PYTHONPATH=rag python -m extensions.adaptive_context_budget run \
  --snapshot "$MLS_RAG_RUN_ROOT/inputs/budget_practice5.jsonl" \
  --output "$MLS_RAG_RUN_ROOT/budget_pilot" \
  --generate --limit 5 --match-adaptive \
  --qrels rag/data/rag_corpus/qrels_practice.tsv \
  --conditions simple_none tfidf_none mmr_none tfidf_fixed_512_skip \
    mmr_fixed_512_skip mmr_adaptive_skip mmr_matched_skip \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/budget_generate.log"

cd "$MLS_DYNAMIC_ROOT"
PYTHONPATH=rag python -m extensions.dynamic_corpus rag-trial \
  --dataset course --generate --limit 5 --k 10 \
  --queries "$MLS_RAG_RUN_ROOT/inputs/practice5_queries.npz" \
  --output "$MLS_RAG_RUN_ROOT/dynamic_trial" \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/dynamic_generate.log"
```

Planned generation calls:

| Investigation | Conditions | Maximum calls |
|---|---|---:|
| Budget | Baseline, TF-IDF, MMR, two fixed treatments, adaptive and matched-cost | 35 |
| Dynamic corpus | Original static, before mutations, after mutations, rebuilt after | 20 |

The dynamic trial shares a generated answer for identical prompts and marks cache
hits. It therefore may make fewer than 20 calls. This proves matched evidence and
avoids redundant generation; it is not an independent generation-determinism test.
Budget bounds use the labelled UTF-8 token estimate unless you supply the actual
generation-model tokenizer. API-reported token usage is recorded separately when
the service returns it.

## 8. Cluster head node: supplied answer judge, then analysis and plots

Judging makes additional real API calls. The shared judge cache reuses identical
question/answer/gold/model judgements. Each condition gets its own copied scores,
so subsequent evaluator invocations cannot overwrite its evidence. On a complete
uncached run there are at most 55 candidate judgements, usually fewer after reuse.

```bash
cd "$MLS_BUDGET_ROOT"
for condition in simple_none tfidf_none mmr_none tfidf_fixed_512_skip mmr_fixed_512_skip mmr_adaptive_skip mmr_matched_skip; do
  python rag/scripts/evaluate_answers.py --split practice --allow-subset --dry-run \
    --candidates "$MLS_RAG_RUN_ROOT/budget_pilot/$condition/candidates.jsonl" \
    2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/budget_${condition}_validate.log" || break
  python rag/scripts/evaluate_answers.py --split practice --allow-subset \
    --candidates "$MLS_RAG_RUN_ROOT/budget_pilot/$condition/candidates.jsonl" \
    --out-dir "$MLS_RAG_RUN_ROOT/shared_judge" \
    2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/budget_${condition}_judge.log" || break
  mkdir -p "$MLS_RAG_RUN_ROOT/budget_pilot/$condition/judge"
  cp "$MLS_RAG_RUN_ROOT/shared_judge/answer_judge_per_query.jsonl" \
    "$MLS_RAG_RUN_ROOT/shared_judge/answer_judge_summary.json" \
    "$MLS_RAG_RUN_ROOT/budget_pilot/$condition/judge/"
done

PYTHONPATH=rag python -m extensions.adaptive_context_budget summarise \
  --run-dir "$MLS_RAG_RUN_ROOT/budget_pilot" \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/budget_summary.log"
PYTHONPATH=rag python -m extensions.adaptive_context_budget plot \
  --run-dir "$MLS_RAG_RUN_ROOT/budget_pilot" \
  --overhead-csv "$MLS_RAG_RUN_ROOT/budget_overhead/overhead.csv" \
  --output "$MLS_RAG_RUN_ROOT/budget_pilot/figures" \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/budget_plot.log"

cd "$MLS_DYNAMIC_ROOT"
for condition in original_static dynamic_before dynamic_after freshly_rebuilt_after; do
  python rag/scripts/evaluate_answers.py --split practice --allow-subset --dry-run \
    --candidates "$MLS_RAG_RUN_ROOT/dynamic_trial/$condition.jsonl" \
    2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/dynamic_${condition}_validate.log" || break
  python rag/scripts/evaluate_answers.py --split practice --allow-subset \
    --candidates "$MLS_RAG_RUN_ROOT/dynamic_trial/$condition.jsonl" \
    --out-dir "$MLS_RAG_RUN_ROOT/shared_judge" \
    2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/dynamic_${condition}_judge.log" || break
  mkdir -p "$MLS_RAG_RUN_ROOT/dynamic_trial/judge_$condition"
  cp "$MLS_RAG_RUN_ROOT/shared_judge/answer_judge_per_query.jsonl" \
    "$MLS_RAG_RUN_ROOT/shared_judge/answer_judge_summary.json" \
    "$MLS_RAG_RUN_ROOT/dynamic_trial/judge_$condition/"
done

PYTHONPATH=rag python -m extensions.dynamic_corpus summarise-trial \
  --output "$MLS_RAG_RUN_ROOT/dynamic_trial" \
  2>&1 | tee "$MLS_RAG_RUN_ROOT/logs/dynamic_summary.log"
```

Only continue to interpretation once all desired conditions have complete judge
outputs. Five queries are a pilot, not strong evidence of preserved quality across
the full practice/dev splits. Cached ASR/encoding means the trial records downstream
costs, not fresh audio-to-answer wall time. Original-qrels/gold interpretation is
limited when a mutation removes relevant information; the controlled indexing
comparison is dynamic-after versus rebuilt-after at the same corpus state.

## 9. Preserve the transferred code identity and retrieve results

The files were copied from uncommitted worktrees; a parent repository's Git HEAD
alone does not identify them. On the cluster, retain source hashes:

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
record = {}
for variable in ("MLS_BUDGET_ROOT", "MLS_DYNAMIC_ROOT"):
    root = Path(os.environ[variable])
    record[variable] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "rag").rglob("*.py"))
        if not any(part in {"data", "models", "__pycache__", ".venv"} for part in path.relative_to(root).parts)
    }
(Path(os.environ["MLS_RAG_RUN_ROOT"]) / "source_sha256.json").write_text(json.dumps(record, indent=2) + "\n")
PY
```

All new run outputs are under:

```text
/home/s2427231/edin-mls-26-autumn/experiments/rag_cluster/runs/practice5_20261112_01/
├── inputs/                 # Real ASR cache, enc_a queries, frozen budget inputs
├── logs/                   # Tests, preparation, benchmark, generation/judge output
├── dynamic_course/         # Raw ablations, exactness checks, summaries, nine figures
├── budget_offline/         # Actual-course packing/token ablations, no model answers
├── budget_offline_adaptive128_1024/
├── budget_offline_adaptive512_2048/
├── budget_overhead/        # Explicitly synthetic controller timing
├── budget_pilot/           # Real candidate answers, usage, scores, paired analysis/figures
├── dynamic_trial/          # Four conditions, evidence/answers/scores, rag_comparison.csv/json
├── shared_judge/           # Reusable judge cache and latest evaluator outputs
└── source_sha256.json
```

Run this **on the Mac**, in a separate local terminal, to copy all new results back:

```bash
mkdir -p /Users/burphh/Desktop/CS/MLS-CW/cluster_results/rag
rsync -avz --progress -e "ssh -J s2427231@ssh.inf.ed.ac.uk" \
  --exclude='matplotlib_cache/' \
  s2427231@mlp.inf.ed.ac.uk:/home/s2427231/edin-mls-26-autumn/experiments/rag_cluster/runs/practice5_20261112_01/ \
  /Users/burphh/Desktop/CS/MLS-CW/cluster_results/rag/practice5_20261112_01/
```

This also copies rather than moves. It does not transfer the shared `.env` key.
