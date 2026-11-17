# Cluster validation, summaries and Excel exports

These commands copy the current local ASR code and both RAG working copies into
the existing cluster checkout, overwriting matching code files. No new worktree
or Git pull is needed. No test, benchmark or API call has been executed locally
for this handoff. Use a fresh results directory to keep each measurement run
internally consistent. The driver stops after a failed stage and still exports
the evidence available.

## 1. Mac: copy the latest local files directly into the cluster checkout

Run in your Mac terminal. The common options exclude machine-specific builds,
credentials and large assets. They do not exclude source, tests, documentation or
the new cluster helpers. The last two commands overlay each latest RAG working
copy onto the snapshot path used by the driver, including uncommitted code.

```bash
CW_LOCAL_ROOT="/Users/burphh/Desktop/CS/MLS-CW/edin-mls-26-autumn-coursework"
CW_REMOTE="s2427231@mlp.inf.ed.ac.uk"
CW_REMOTE_ROOT="/home/s2427231/edin-mls-26-autumn"
CW_SYNC_ARGS=(
  -avz --progress -e "ssh -J s2427231@ssh.inf.ed.ac.uk"
  --exclude='.git' --exclude='.env*'
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/'
  --exclude='.cache/' --exclude='.mplconfig/' --exclude='.venv/'
  --exclude='venv/' --exclude='env/' --exclude='build/'
  --exclude='*.o' --exclude='*.so' --exclude='.DS_Store'
  --exclude='data/' --exclude='models/' --exclude='results/'
  --exclude='rag-assets.zip'
)

rsync "${CW_SYNC_ARGS[@]}" \
  --exclude='/rag/rag-adaptive-context-budget/' \
  --exclude='/rag/rag-dynamic-corpus/' \
  "$CW_LOCAL_ROOT/" "$CW_REMOTE:$CW_REMOTE_ROOT/"

rsync "${CW_SYNC_ARGS[@]}" \
  "$CW_LOCAL_ROOT/rag/rag-adaptive-context-budget/rag/" \
  "$CW_REMOTE:$CW_REMOTE_ROOT/experiments/rag_snapshots/rag-adaptive-context-budget/rag/"

rsync "${CW_SYNC_ARGS[@]}" \
  "$CW_LOCAL_ROOT/rag/rag-dynamic-corpus/rag/" \
  "$CW_REMOTE:$CW_REMOTE_ROOT/experiments/rag_snapshots/rag-dynamic-corpus/rag/"

ssh -J s2427231@ssh.inf.ed.ac.uk "$CW_REMOTE"
```

Matching cluster code files are overwritten. Local files remain in place. No Git
commit/push is required: rsync transfers the working files directly. The cluster's
Git metadata, configured `.env`, installed data/models and existing binaries are
excluded from transfer; the ASR stage rebuilds the binaries on its allocated GPU.

## 2. Do the teammate changes need pulling?

**No additional pull is required to run the saved experiments.** Both snapshots
already contain the teammate's TF-IDF, MMR and reranker at `e3ed9ea`. Top-level
`rag/src/` is an older baseline; the driver deliberately uses the snapshots.

If you want to inspect whether your mate has made newer relevant changes, run
this from the cluster checkout after copying the code:

```bash
cd /home/s2427231/edin-mls-26-autumn
git fetch https://github.com/awg2024/edin-mls-26-autumn-coursework.git feature/rag_retrieval_context
git log --oneline e3ed9ea..FETCH_HEAD -- rag/src rag/scripts
git diff --stat e3ed9ea FETCH_HEAD -- rag/src rag/scripts
```

Empty output means no changes on those paths relative to that snapshot. Nonempty
output requires review and deliberate integration into each experimental runtime
before claiming to benchmark the latest teammate version. A generic `git pull`
does not update copied snapshot directories and may create unrelated conflicts.
New remote commits have not been verified from this local handoff.

## 3. Head node: environments, assets and credentials

Set these once in the head shell; `srun` normally inherits exported variables:

```bash
export CW_ROOT="/home/s2427231/edin-mls-26-autumn"
export CW_RUN="$HOME/mls-results/direct-20261115-01"
export CW_ASSETS="/home/s2427231/edin-mls-26-autumn/rag"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda info --envs
export RAG_ASR_PYTHON="$(conda run -n mls-asr python -c 'import sys; print(sys.executable)')"

if [ ! -f "$HOME/mls-rag-env/bin/activate" ]; then
  conda run -n mls-asr python -m venv "$HOME/mls-rag-env"
fi
source "$HOME/mls-rag-env/bin/activate"
python -m pip install --no-cache-dir "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir "torch==2.10.0+cpu" \
  -r "$CW_ROOT/experiments/rag_snapshots/rag-adaptive-context-budget/rag/requirements.txt" \
  -r "$CW_ROOT/experiments/rag_snapshots/rag-adaptive-context-budget/rag/extensions/adaptive_context_budget/requirements-experiments.txt" \
  -r "$CW_ROOT/experiments/rag_snapshots/rag-dynamic-corpus/rag/extensions/dynamic_corpus/requirements.txt"
```

The RAG query encoder/reranker runs on CPU in these experiments. Installing its
CPU PyTorch wheel first and requiring that exact build prevents an unconstrained
sentence-transformers installation from pulling a separate CUDA stack. ASR keeps
its CUDA PyTorch in `mls-asr`, selected through `RAG_ASR_PYTHON`. Install packages
on the head/login node, not inside a time-limited GPU allocation.

If a previous installation was killed halfway through, recreate only the RAG
venv at the same path before the pip commands above:
`"$RAG_ASR_PYTHON" -m venv --clear "$HOME/mls-rag-env"`.
This removes that venv's incomplete packages; source/data/results and the ASR
environment are outside it.

If `mls-asr` is missing or its setup/model download is incomplete, first run
`bash /home/s2427231/edin-mls-26-autumn/hw1-asr/setup_env.sh` on the head node,
then repeat the interpreter/venv commands. Compute nodes may lack download access.

Reuse installed corpus/audio/vectors under `$CW_ASSETS/data/rag_corpus/` and the
real query model under `$CW_ASSETS/models/enc_a/`. If they are missing, use the
original checkout's `rag/README.md` asset-install steps (Git LFS archive, extraction
and `rag/scripts/install_assets.py`). No assets are modified by the experiments.

For real API calls, edit the shared configuration:

```bash
cd /home/s2427231/edin-mls-26-autumn
if [ ! -f rag/.env ]; then
  cp rag/.env.example rag/.env
fi
chmod 600 rag/.env
nano rag/.env
```

Set your ELM key and the assigned endpoint/model. The driver reads this file for
the explicit `generate`/`judge` stages; it does not print or export the key to a
results file. No symlinks or copying credentials into experiment folders is needed.

## 4. GPU node: validate all ASR tasks and the complete model

```bash
srun -p Teaching --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

conda run --no-capture-output -n mls-asr \
  python "$CW_ROOT/tools/cluster/run.py" asr
```

This runs, in `hw1-asr`, on the same allocated GPU:

```text
nvidia-smi -L
nvcc --version
make clean
make
make test
bash run.sh --cuda none --profile
bash run.sh --cuda all --profile
```

`make test` invokes the original four supplied test executables exactly once.
CUDA-all transcription runs only after both the complete kernel suite and the
PyTorch reference succeed. A failed task remains a failed task; no source/tests
are altered or hidden behind a fallback. The independent PyTorch control still
runs when kernels fail, if compilation completed, to help separate kernel defects
from environment/model issues.

Look at:

```bash
cat "$CW_RUN/SUMMARY.md"
cat "$CW_RUN/excel/asr_checks.csv"
```

Success requires Task 1–4 **ALL PASS**, plus both model modes **Status: PASS**.
The benchmark's printed accuracy is expected-word-set coverage on the supplied
single audio sample, not a WER benchmark. Inspect the saved transcription too.

Raw output is in `$CW_RUN/asr/`; the build, tests, two model profiles and exit codes
are retained. Failures stop the driver with a nonzero exit code and a partial
summary. Preserve them; use a fresh `CW_RUN` for another full attempt.

If a task fails, run only its diagnostics into another log. Example for Task 4:

```bash
cd "$CW_ROOT/hw1-asr"
set -o pipefail
make test4 2>&1 | tee "$CW_RUN/asr/task4_diagnostic.log"
```

If the kernel suites pass but CUDA-all transcription fails, investigate with
`ASR_DEBUG=1 bash run.sh --cuda all --profile` and/or cumulative task groups, saving
each diagnostic log. Do not treat a failed integration run as a passing result.

## 5. Same allocation: RAG tests, real inputs and non-API ablations

After ASR validation passes:

```bash
source "$HOME/mls-rag-env/bin/activate"
python "$CW_ROOT/tools/cluster/run.py" prepare
python "$CW_ROOT/tools/cluster/run.py" offline
```

If you already ran the README's ASR commands manually and have the three passing
logs in `hw1-asr/`, do not repeat them. Use a fresh `CW_RUN` and replace the
`prepare` invocation with:

```bash
python "$CW_ROOT/tools/cluster/run.py" prepare \
  --validated-asr-logs "$CW_ROOT/hw1-asr"
```

This imports `validation_tests.log`, `validation_pytorch.log` and
`validation_cuda_all.log`, verifies all four task summaries/model PASS markers and
their kernel settings, and archives their hashes. Missing clean-build output,
historical source hashes and shell exit codes remain explicitly unrecorded.
Only RAG tests/input preparation run now. The `generate` and `judge` stages may
run before `offline` if you want real API results first; the overhead plot is
omitted until its separately measured CSV exists.

`prepare` runs both Python test suites, transcribes the first five practice
queries using the validated CUDA-all ASR, saves a shared transcript/query cache,
and freezes budget inputs with Simple/TF-IDF/MMR builders at k=5. Dynamic retrieval
uses fixed k=10. The query encoder stays on CPU. All original doc/vector ID mapping
and qrels are preserved. Existing output files are not overwritten.

If intentionally investigating RAG independently of incomplete ASR kernels, you
may instead run `prepare --asr-cuda none`; this choice is recorded. It does not
make an ASR failure pass. Use the same ASR configuration for every matched RAG arm.

`offline` runs all 21 dynamic cases (three strategies × three repeats) with actual
enc_a corpus vectors and those real query embeddings, including batch/type/
accumulation/threshold/read-ratio/corpus-size ablations and nine PNG/PDF figures.
It also runs budget sizes 256/512/1024/2048, both packing policies, matched cost,
two additional adaptive ranges, and a labelled synthetic controller-overhead
microbenchmark. These stages make **zero ELM calls**.

Then release the GPU:

```bash
exit
```

The indexing and packing calculations are CPU workloads; being on a GPU node does
not turn them into CUDA benchmarks. Their manifests record dataset, dimensions,
query counts, workload settings and timing scope.

## 6. Head node: real generation, supplied judging and all summaries

Back in the same head-node shell, keep the same `CW_*` variables. If using another
SSH session, export the three paths from section 3 again. Then:

```bash
source "$HOME/mls-rag-env/bin/activate"
python "$CW_ROOT/tools/cluster/run.py" generate
python "$CW_ROOT/tools/cluster/run.py" judge
python "$CW_ROOT/tools/cluster/run.py" summary
cat "$CW_RUN/SUMMARY.md"
```

These reuse the real cached inputs, so no GPU or live ASR is needed. Head-node API
execution avoids depending on outbound connectivity from compute nodes.

Generation is 35 planned calls for seven budget arms (five queries each) and at
most 20 for four dynamic corpus conditions. The dynamic trial shares answers for
identical prompts and labels cache hits. Judging uses the existing evaluator,
validates candidate schemas, and shares its cache while copying per-condition
scores to the locations expected by each analysis tool. At most 55 uncached
candidate judgements are planned. This is a pilot, not a full 350-query API sweep.

For the four-point **MMR quality vs budget** chart, use this instead of the
plain `generate` command on the first attempt:

```bash
python "$CW_ROOT/tools/cluster/run.py" generate --budget-sweep
```

It adds MMR budgets 256, 1024 and 2048 alongside the existing 512 arm: 15 additional
generation calls and up to 15 additional judgements. Total planned generation is
then 50 budget calls + at most 20 dynamic calls. The `judge` stage automatically
reads the actual generated arm list, so its command is unchanged. Do not execute
both generation commands into the same output directory. This is an explicit
fixed-budget ablation over five queries, not a retrieval K sweep.

Budget output records actual API token usage where returned, plus separately
labelled estimates, controller latency, generation latency and rubric scores.
Dynamic trial output records evidence/IDs, answers, cached-ASR downstream timing,
cache flags and judge scores. None of the cached-input latencies should be called
fresh audio-to-answer wall time.

Each stage is explicit; `summary` never calls ELM or reruns a benchmark. It can be
used on partial runs. Missing measurements are blank/Not measured, not zero.

## 7. Files for Excel and the report

Open CSVs using Excel **Data → From Text/CSV**, UTF-8. The exporter adds a UTF-8
BOM, so ordinary direct opening generally works too. Use a separate worksheet
for each scope; keep the teammate's 350-query K-sweep sheet as its own experiment.

| Worksheet / CSV under `$CW_RUN/excel/` | Suggested columns | Suggested graph |
|---|---|---|
| `asr_checks.csv` | Task, PASS/FAIL, case count, failed cases, max error | Correctness table |
| `asr_models.csv` | Mode, transcription mean/SD ms, ms/token, peak GiB, word-set coverage, speedup | PyTorch vs CUDA latency bars |
| `asr_kernel_cases.csv` | Task, case/shape, max error, kernel ms, GB/s or GFLOP/s; Task 3 naive/cuBLAS rates | Compare representative shapes within a task |
| `asr_profiles.csv` | Mode, operation, backend, calls, total kernel ms, kernel share % | Stacked operation costs |
| `dynamic_systems.csv` | Family, strategy, source/active N, D, operations, batch, query ratio, threshold, repetitions, insert/update/delete ms, maintenance ms/SD, query mean/p95, matrix MiB, speedup, exact agreement | Batch size vs amortised update cost; threshold vs query time; query ratio vs workload |
| `dynamic_retrieval.csv` | Case/strategy, final-state scope, unique queries/repeats, original Recall@10, active Recall/MRR/NDCG@10, relevant-document availability | Availability vs recall; same-state strategy comparison |
| `budget_conditions.csv` | Run/condition, selector/policy, query counts, generated/judged coverage, evidence before/after, reduction %, actual API tokens, controller/generation ms, quality/retention, matched-cost error | Token count vs quality; tokens vs generation latency |
| `budget_paired.csv` | Baseline/treatment, comparison type, metric, paired N/coverage, delta, bootstrap interval | Paired quality/cost changes |
| `dynamic_answers.csv` | Condition, query count, cache hits, judged count, retrieval/downstream means, uncached generation mean, correctness/grounding/completeness/normalised score | Dynamic-after vs rebuilt-after evidence/quality table |

For dynamic update-vs-batch graphs, the original
`dynamic_course/summary.csv` also contains `amortised_update_ms_mean` and SD,
updates/s and separate staging/rebuild/compaction times. Raw state history supports
accumulation and memory graphs. The driver reuses existing report plots rather
than recreating the teammate's K sweep.

Keep these distinctions explicit:

- Five matched pilot queries are not the 350 queries in the screenshot. Do not
  label this pilot as a 350-query evaluation or directly attribute score differences
  against that sheet to these new components.
- Budgeting follows retrieval. For a fixed selector/ranking, the evidence and answer
  changes are the relevant comparison; it does not create a new retrieval K curve.
- Dynamic retrieval must match a rebuilt index **at the identical active corpus
  state**. Relevant documents deliberately removed from the corpus affect
  availability, not indexing correctness. Original and active qrels scopes differ.
- Fraction columns are 0–1: use Excel Percentage formatting. Fields ending `_pct`
  are already 0–100: use Number formatting, avoiding a second multiplication by 100.
- Embedding MiB is not total RSS; mean per-repeat p95 is not a pooled p95.
- A configured evidence budget is not actual API input tokens. The default counter
  estimates UTF-8 bytes/4 and excludes some prompt overhead.

`SUMMARY.md` is the readable results summary. Original modules also produce
`dynamic_course/principal_table.md`, `summary.csv/json`, correctness JSON, raw
logs/timings and plots; `budget_pilot/summary.csv`, paired summaries and figures;
and `dynamic_trial/rag_comparison.csv/json`. `source_sha256.json` identifies the
actual code, including locally transferred helpers; Git HEAD alone is insufficient.

## 8. Copy all results back to the Mac

On the **Mac**:

```bash
mkdir -p /Users/burphh/Desktop/CS/MLS-CW/cluster_results
rsync -avz --progress -e "ssh -J s2427231@ssh.inf.ed.ac.uk" \
  --exclude='matplotlib_cache/' \
  s2427231@mlp.inf.ed.ac.uk:/home/s2427231/mls-results/direct-20261115-01/ \
  /Users/burphh/Desktop/CS/MLS-CW/cluster_results/direct-20261115-01/
```

Both copies remain. No credential file or large course assets are in this result
directory. If a stage fails, retain the summary and its last log for debugging.
