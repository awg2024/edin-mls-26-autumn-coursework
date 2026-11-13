# Adaptive evidence-budget control

Can controlling **how much already-ranked evidence** reaches ELM reduce token use
and generation latency while retaining correctness, grounding and completeness?
This extension changes context amount, not retrieval or sentence ranking.

## Branch and existing components

Implementation branch: `feat/rag-adaptive-context-budget`, in the isolated
`rag-adaptive-context-budget/` worktree. Its base is teammate branch
`origin/feature/rag_retrieval_context` at `e3ed9ea`.
Use that branch as the PR base until its work is merged into main.

The supplied `SimpleContextBuilder`, teammate `ExperimentalContextBuilder`,
`select_tfidf`, `select_mmr`, sentence utilities and `CrossEncoderReranker` are
reused **without algorithm changes**. Their existing 5000-character selection cap
and MMR lambda=0.7 remain controls. We do not run another K-sweep; retrieval K is
fixed at the coursework default 5 unless a teammate-selected value is supplied.
The original CUDA checkout and its uncommitted changes are separate.

## Controller and token contract

- `core.py`: ordered `EvidenceUnit` objects, token counters, fixed/adaptive
  decisions, and packing. Accepted units keep their original order and labels.
- `integration.py`: wraps the existing builder; preserves its exact question and
  instruction prefix. `policy=None` returns its original dictionary unchanged.
- `experiment.py`: freezes candidate retrieval and existing selector outputs
  once, then replays budget conditions over those same inputs.
- `analysis.py`: joins outputs from the supplied answer judge, calculates paired
  differences/coverage and creates figures only when their measurements exist.

The default count is **ceil(UTF-8 bytes / 4)** over the complete evidence string,
including document labels and separators. This is a deterministic **estimate**,
not the ELM tokenizer and not an API context-limit guarantee. A configured
budget is strictly enforced against this counter. To enforce a tokenizer-based
bound, supply `--tokenizer-path` pointing to a local copy of the **generation
model's** tokenizer. The enc_a retrieval tokenizer is not a substitute.

Evidence budgets exclude the question, instructions and chat wrapper. Estimated
prompt message-text tokens additionally count the user and system text; chat
template overhead is excluded and explicitly labelled. API-reported input,
output and total usage are stored separately when ELM returns them.

Fixed packing offers:

1. `skip`: skip an oversized unit and continue in the existing order.
2. `sentence_prefix`: keep its longest fitting complete-sentence prefix, or skip
   when none fits. No arbitrary word cuts, relevance scoring or reranking.

First-occurrence exact-text deduplication avoids repeated evidence and logs
duplicate/oversized discards. Existing TF-IDF/MMR already deduplicate sentences.
The no-budget controls are never deduplicated or otherwise changed by this layer.
Sentence-prefix packing mainly helps whole-document units; a unit that is already
one sentence may offer no additional packing opportunity.
Prefix boundaries use a punctuation heuristic, which can misidentify
abbreviations; this is not a language-aware sentence parser.

## Adaptive policy

For at least three finite retrieval scores, compute:

```text
confidence = (largest - second_largest) / (largest - smallest)
fraction   = clamp((confidence - low) / (high - low), 0, 1) ** gamma
budget     = round(max_budget - fraction * (max_budget - min_budget))
```

Defaults: min=256, max=2048, low=0.1, high=0.6, gamma=1. Large relative top-score
gaps favour smaller contexts; ambiguous/tied scores favour larger ones. The
statistic is invariant to positive affine score changes, avoids absolute cosine
thresholds, and does not consume qrels, reference answers or judge outputs.
Missing, invalid or fewer than three scores conservatively use the maximum.
Empty evidence uses the minimum allocated budget and zero actual tokens.

This is a retrieval-ambiguity heuristic, not calibrated answer confidence. A
large retrieval gap can still accompany a difficult or incorrectly retrieved
question. Quality at **matched used-token cost** must establish whether it helps.

## Live pipeline / CLI

The existing answer-generation CLI keeps its defaults. Budgeting is opt-in:
For real audio runs, first configure `RAG_ASR_PYTHON` to the ASR environment's
interpreter as in `rag/README.md`. Use `ASR_CUDA=none` for the supplied PyTorch
reference until validated CUDA kernels have been integrated into this worktree.

```bash
python rag/scripts/generate_answers.py --split practice --limit 5 \
  --context-strategy mmr --context-budget fixed --evidence-token-budget 512 \
  --budget-packing-policy skip --output rag/results/mmr_fixed512_answers.jsonl
```

Use `--context-budget adaptive --adaptive-min-budget 256 --adaptive-max-budget 2048`
for adaptive control. With `--context-budget none` the original builder and
default generator output remain unchanged. Prefer the frozen-input runner below
for paired experiments; separate live CLI invocations would repeat ASR/retrieval.

## Environment and tests

Run commands from this worktree's repository root, using the RAG environment:

```bash
python -m pip install -r rag/requirements.txt
python -m pip install -r rag/extensions/adaptive_context_budget/requirements-experiments.txt
export PYTHONPATH="$PWD/rag${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s rag/tests -v
```

Tests mock ASR and ELM but use the actual existing sentence selectors. They cover
budget boundaries/formatting, Unicode, duplicates, order, provenance, empty
contexts, safe truncation, adaptive fallback/ties, unchanged disabled behaviour,
paired-input identity, supplied evaluator schema, cost matching, coverage, plots
and API safeguards. Mock results are never answer-quality evidence. This task's
explicit request authorises local CPU tests; no CUDA tests are run here.

## Prepare controlled course inputs once

Install course assets using `rag/README.md`, and create/reuse the existing ASR
transcript cache. GPU ASR is needed only if the cache has not been created.
Use the same ASR configuration/cache for every condition. This RAG worktree is
based on the teammate's branch, so its CUDA starter files are not evidence of
the original checkout's implemented kernels.

```bash
export ASR_CUDA=none
python rag/scripts/evaluate_rag.py --device cpu --split practice \
  --transcript-cache rag/.cache/practice_transcripts.jsonl

python -m extensions.adaptive_context_budget prepare \
  --split practice --limit 5 --k 5 \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --output rag/results/budget_snapshots/practice5.jsonl
```

Preparation reuses cached transcripts, one embedding/retrieval per query, and one
call to each selected existing builder. It saves prompts, selected evidence,
scores, timings, provenance hashes and a `.lengths.csv` inspection table.
Missing transcripts fail explicitly; no mock transcript fallback is used.
An optional `--reranker-model /local/path --final-k 5` freezes the existing
cross-encoder's output for all subsequent conditions.

Inspect `.lengths.csv` **before** choosing budgets. A 2048-token estimate may
already exceed the teammate selector's 5000-character cap; a no-op budget is a
saturation control, not evidence of a useful optimisation.

## Offline token and packing ablations

```bash
python -m extensions.adaptive_context_budget run \
  --snapshot rag/results/budget_snapshots/practice5.jsonl \
  --budgets 256 512 1024 2048 --packing-policies skip sentence_prefix \
  --match-adaptive --qrels rag/data/rag_corpus/qrels_practice.tsv \
  --output rag/results/budget_offline_practice5
```

This makes **zero ELM calls**. Conditions include:

| Coursework comparison | Conditions |
|---|---|
| A | `simple_none`, original baseline |
| B versus D | `tfidf_none` versus `tfidf_fixed_512_skip` (and other fixed budgets) |
| C versus E | `mmr_none` versus `mmr_fixed_512_skip` |
| C versus F | `mmr_none` versus `mmr_adaptive_skip` |
| Matched-cost check | `mmr_matched_skip` versus `mmr_adaptive_skip` |

Each packing policy has separate conditions. `--match-adaptive` searches a fixed
budget giving approximately the adaptive mean **used evidence tokens** on this
snapshot. It uses lengths/scores only, not labels. Check `matched_cost_error_pct`
in `summary.csv`; discrete sentence sizes can prevent an exact match. Declare a
cost tolerance (e.g. 5%) before claiming a matched-cost quality advantage.
For dev validation, carry parameters selected on practice rather than tuning to
dev quality. Matching the same snapshot is a descriptive cost control, not an
independently trained or validated policy.

For the parameter ablation, repeat into fresh output directories with, for
example, `--min-budget 128 --max-budget 1024`, then
`--min-budget 512 --max-budget 2048`, or vary `--high-confidence`/`--gamma` one at
a time. Keep the snapshot, selectors, K, model and generation parameters fixed.

## Small real ELM pilot and supplied evaluation

Configure the existing `rag/.env` first. Generate five matched queries per arm:

```bash
python -m extensions.adaptive_context_budget run \
  --snapshot rag/results/budget_snapshots/practice5.jsonl \
  --output rag/results/budget_pilot --generate --limit 5 --match-adaptive \
  --conditions simple_none mmr_none mmr_fixed_512_skip mmr_adaptive_skip mmr_matched_skip
```

This explicitly requests 25 generation calls. To include the TF-IDF paired pilot,
use a separate run with `--conditions tfidf_none tfidf_fixed_512_skip` (10 calls).
The runner retains the supplied model, temperature=0.1 and system instructions.
It does not judge answers or automatically generate across every ablation.

Evaluate each generated condition with the **unchanged coursework evaluator**:

```bash
for condition in simple_none mmr_none mmr_fixed_512_skip mmr_adaptive_skip mmr_matched_skip; do
  python rag/scripts/evaluate_answers.py --split practice --allow-subset --dry-run \
    --candidates "rag/results/budget_pilot/$condition/candidates.jsonl"
  python rag/scripts/evaluate_answers.py --split practice --allow-subset \
    --candidates "rag/results/budget_pilot/$condition/candidates.jsonl" \
    --out-dir "rag/results/budget_pilot/$condition/judge"
done
python -m extensions.adaptive_context_budget summarise --run-dir rag/results/budget_pilot
python -m extensions.adaptive_context_budget plot --run-dir rag/results/budget_pilot \
  --output rag/results/budget_pilot/figures
```

Judging makes further ELM calls and reuses the supplied judge cache/rubric.
Use `--repeats` on that evaluator for repeated judgements where appropriate.
To expand to practice, prepare with `--limit 350`, then explicitly select
conditions with `run --generate --limit 350 --allow-large-api-run` and fresh
output paths. For all 125 dev queries, prepare with `--split dev --limit 125` and
use dev qrels/evaluation. Broader API sweeps are never automatic.

## Outputs, uncertainty and figures

Every condition has `metrics.jsonl`/`.csv`; real generation additionally creates
evaluator-compatible `candidates.jsonl`. Offline runs create no fake answers.
The manifest records frozen inputs and policy parameters. Summaries report
sample counts, mean/sample SD, quality coverage, API usage when returned, and
paired per-query changes with deterministic 2000-resample bootstrap intervals.
Paired summaries label same-selector effects as `budget_only`, matched-cost
effects as `budget_only_matched_cost`, and comparisons against the original
Simple builder as `selector_and_budget_practical`; the latter also include the
effect of sentence selection and must not be attributed solely to budgeting.
Intervals are unavailable for fewer than two paired observations; small pilots
are diagnostics, not strong evidence of equivalence.

`query_groups.csv` and `paired_groups.csv` stratify by confidence and selected
budget. Confidence bins use the same frozen retrieval statistic for controls and
treatments. Qrels are used only after packing to measure relevant-document
retention via existing metric utilities. They do not prove that any particular
selected sentence contains the answer.

Preparation-stage ASR is cached. `reconstructed_downstream_ms` combines recorded
shared-stage costs with controller/generation timing; it is not a newly measured
end-to-end wall time. `budget_and_generation_wall_ms` is measured directly.
Live `RAGPipeline` runs still expose actual end-to-end timing. Network-dependent
generation latency and local controller cost remain separately labelled.

The plotting command supports all six requested figures in PNG and PDF. Missing
quality/generation data are listed in `plot_manifest.json`; no placeholder score
curves are drawn. Compare the best budgeted condition separately with
`simple_none`, while attributing B/D, C/E and C/F differences to budget control.

## Synthetic overhead microbenchmark

```bash
python -m extensions.adaptive_context_budget overhead \
  --output rag/results/budget_overhead --counts 1 5 20 100 500 1000 --repeats 30
python -m extensions.adaptive_context_budget plot \
  --overhead-csv rag/results/budget_overhead/overhead.csv \
  --output rag/results/budget_overhead/figures
```

This varies already-ranked evidence count, not retrieval K. Both policies use
1024-token bounds to hold packing work constant; adaptive confidence computation
still executes. Warmups=5, timing excludes input construction, and tracemalloc
peak allocations are measured separately from timed repetitions. Results are
synthetic CPU costs, not evidence of course answer quality or ELM speedup.

See [report_findings.md](report_findings.md) and the saved
[actual local measurements](measurements/local_cpu_20261111/overhead.csv).
