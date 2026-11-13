# Budget-aware and adaptive context: findings

## Research question and hypothesis

Can an evidence-token cap and a retrieval-confidence-based adaptive cap reduce
ELM input cost while preserving correctness, grounding and completeness?

The hypothesis is that strongly separated retrieval scores often need less
evidence, while ambiguous retrieval benefits from a larger allocation. This is a
testable heuristic, not an established relationship or calibrated confidence.

**Current conclusion:** implementation and mock integration checks pass.
Synthetic CPU controller overhead has been measured. Course answer quality,
real token savings and ELM latency changes remain **unmeasured**, because the
course assets, cached transcripts and credentials are absent.

## Contribution and repository state

The implementation is on `feat/rag-adaptive-context-budget` in the isolated
`/Users/burphh/Desktop/CS/MLS-CW/rag-adaptive-context-budget` worktree, based on
teammate branch `origin/feature/rag_retrieval_context` at `e3ed9ea`.

The original checkout remains on `feature/task4_attention`; its CUDA and
`rag/.env.example` edits were preserved. No commits or pushes were created.

New code under this extension provides:
- A reusable ordered evidence/provenance interface and explicit token counter.
- Fixed and query-adaptive budget policies.
- Skip and complete-sentence-prefix packing with exact-text deduplication.
- A builder wrapper with a transparent disabled path.
- Frozen-input experiments, used-token cost matching, per-query CSV/JSONL,
  summaries, paired bootstrap intervals, stratification and PNG/PDF plotting.
- Synthetic packing-cost measurements and unit/integration tests.

Only three existing files need integration changes:
`rag/src/rag.py` (optional retrieval-aware builder hook and enabled-only metrics),
`rag/src/generator.py` (opt-in API token-usage capture with identical requests),
and `rag/scripts/generate_answers.py` (optional budget flags).

## Existing components reused

TF-IDF, MMR, their sentence/provenance utilities, the cross-encoder reranker,
`SimpleContextBuilder`, retrieval, encoder, metric utilities and the supplied
ELM judge are reused. Their ranking/selection algorithms are unchanged. The
implementation does not repeat the teammate's retrieval K-sweep.

TF-IDF/MMR already select sentences subject to a 5000-character cap; the new cap
is applied **after that existing selection**, rather than assuming that all
retrieved documents enter the prompt. MMR lambda remains 0.7 by default.
The original Simple builder receives no additional constraint when disabled.

## Baselines and experimental controls

| Arm | Selector | Additional budget |
|---|---|---|
| A | Original SimpleContextBuilder | None |
| B | Existing TF-IDF | None |
| C | Existing MMR | None |
| D | Same TF-IDF output as B | Fixed |
| E | Same MMR output as C | Fixed |
| F | Same MMR output as C | Adaptive |
| Matched-cost control | Same MMR output as F | Fixed, calibrated to approximately the adaptive mean used evidence tokens |

B/D, C/E and C/F isolate budget control. Comparisons against A also include the
effect of sentence selection and are labelled `selector_and_budget_practical`.
The analysis verifies that paired rows have the same transcript and candidate
ranking; budget-only pairs must also have the same selector-output hash.

Preparation uses one cached ASR transcript, one retrieval (default K=5), and one
call per existing selector per query. Optional reranking is frozen at this stage.
Every condition reuses these outputs. Question/system instructions, generation
model and temperature=0.1 remain fixed. Qrels enter evaluation after packing,
never the adaptive policy.

Matching uses **actual used tokens under the selected counter**, not just the
configured budget. The fixed/adaptive mismatch is recorded; a proposed 5%
tolerance should be declared before interpreting a matched-cost quality claim.
Use practice for parameter selection and dev for additional validation.

## Automated validation actually executed

Command, from the worktree root:

```bash
/opt/miniconda3/envs/diss/bin/python -m unittest discover -s rag/tests -v
```

**57 tests passed** (9.216 seconds in the archived final run), including the
three existing ASR adapter tests. Source:
[test_results.txt](measurements/local_cpu_20261111/test_results.txt).

Coverage includes exact formatting overhead and separators, boundary budgets,
duplicates, empty/oversized units, Unicode, provenance, deterministic order,
safe truncation, score ties/missing scores, adaptive boundaries, unchanged
disabled Simple/TF-IDF/MMR output, actual pipeline integration with mocks, frozen
candidate identity, the supplied answer-candidate validator, API-usage capture,
matched used-token cost, partial judge coverage, bootstrap output, and CLI
prepare-to-offline-run integration.

All six plot types were exercised using **temporary mock fixtures** and those
test figures were deleted. They are not published as measurements. The tests
prove protocol/implementation properties, not improved answer quality.

Local CPU execution follows the explicit request in this RAG task. CUDA tests
were not run.

## Actual overhead and scalability measurements

Data:
[aggregate CSV](measurements/local_cpu_20261111/overhead.csv),
[720 individual timings](measurements/local_cpu_20261111/overhead_samples.csv),
[method/machine manifest](measurements/local_cpu_20261111/benchmark_manifest.json).

Host: macOS-26.5.1-arm64-arm-64bit.
Python: 3.11.11 | packaged by conda-forge | (main, Dec  5 2024, 08:47:03) [Clang 18.1.8 ].

This is **synthetic ranked evidence**, not course documents or retrieval
quality. Counts are 1, 5, 20, 100, 500 and 1000 units; four controller/packing
configurations give 24 measured cases. Each has five warmups and thirty timed
calls. Both fixed and adaptive policies use 1024-token bounds so packing work is
controlled; adaptive still computes the retrieval statistic. Input construction
is excluded. Timing includes decision, full formatted-string counting, packing
and metrics. Separate tracemalloc calls measure additional Python allocations,
not total process RSS.

Selected rows (all 24 cases are in the CSV):

| Configuration | Ranked units | Mean ms | Sample SD ms | P95 ms | Peak Python KiB |
|---|---:|---:|---:|---:|---:|
| fixed_skip | 5 | 0.0145 | 0.0008 | 0.0162 | 6.7 |
| fixed_sentence_prefix | 5 | 0.0131 | 0.0003 | 0.0137 | 6.7 |
| adaptive_skip | 5 | 0.0139 | 0.0002 | 0.0143 | 6.8 |
| adaptive_sentence_prefix | 5 | 0.0140 | 0.0006 | 0.0143 | 6.8 |
| fixed_skip | 100 | 0.1631 | 0.0587 | 0.3190 | 51.9 |
| fixed_sentence_prefix | 100 | 0.4162 | 0.0174 | 0.4644 | 62.8 |
| adaptive_skip | 100 | 0.1418 | 0.0106 | 0.1690 | 51.9 |
| adaptive_sentence_prefix | 100 | 0.4195 | 0.0175 | 0.4648 | 62.8 |
| fixed_skip | 1000 | 1.0737 | 0.0174 | 1.1191 | 442.0 |
| fixed_sentence_prefix | 1000 | 4.7641 | 0.2624 | 5.1624 | 472.1 |
| adaptive_skip | 1000 | 1.1211 | 0.1950 | 1.1157 | 442.1 |
| adaptive_sentence_prefix | 1000 | 4.6666 | 0.0603 | 4.7245 | 473.0 |

For this fixture, skip packing is about 1.1 ms at 1000 units; sentence-prefix
packing is about 4.7 ms. Prefix search requires additional boundary/count trials.
Both costs increase with evidence count. Small fixed/adaptive timing differences
are not evidence of a general speed advantage; they require repetition on
representative inputs. Thirty repetitions of a fixed synthetic input do not
represent thirty independent query workloads.

The measured overhead alone cannot establish whether an ELM latency reduction
would exceed controller cost. That comparison needs real generation timing.

## Real course quality/cost results

| Method | Generated queries | Judged queries | API input tokens | Correctness / grounding / completeness | Answer score | Generation latency |
|---|---:|---:|---|---|---|---|
| Original baseline | 0 | 0 | Unmeasured | Unmeasured | Unmeasured | Unmeasured |
| TF-IDF without / with budget | 0 | 0 | Unmeasured | Unmeasured | Unmeasured | Unmeasured |
| MMR without / fixed / adaptive budget | 0 | 0 | Unmeasured | Unmeasured | Unmeasured | Unmeasured |

There is no measured baseline-versus-budgeted answer-quality, quality-retention,
token-reduction or network-latency improvement to report.

[resources.json](measurements/local_cpu_20261111/resources.json) records:
- Corpus, audio manifest, embeddings and enc_a weights absent.
- Cached ASR transcripts absent.
- `rag-assets.zip` is a 134-byte Git LFS pointer, not an installed asset archive.
- `rag/.env` and process ELM credentials absent; the local test interpreter also
  lacks the OpenAI SDK used by ELM.
- **Zero real generation calls and zero real judge calls.**

An actual course-input preparation attempt failed explicitly with exit status 2
for missing inputs; see
[course_prepare_blocked.txt](measurements/local_cpu_20261111/course_prepare_blocked.txt).
Synthetic text was not substituted into a course-quality evaluation.

## Ablation findings and remaining execution

| Ablation | Implemented / observed | Remaining |
|---|---|---|
| Fixed budgets 256/512/1024/2048 | Runner and boundary tests available | Inspect real length CSV and evaluate quality/latency curves |
| Fixed vs adaptive | Same-input matching and paired analysis tested | Evaluate adaptive against approximately equal used-token fixed controls |
| Adaptive parameters | Min/max, gap thresholds and gamma configurable and tested | Small controlled practice parameter study, then dev validation |
| Skip vs sentence prefix | CPU overhead measured; correctness tested | Course evidence utilisation, retained document coverage and quality |
| Overhead vs candidate count | 24 configurations, 720 timing samples, memory allocation measurements | Repeat on representative course evidence and deployment host |
| Query characteristics | Shared confidence strata and paired quality-group CSV implemented | Identify actual groups that lose quality under reduction |

The 2048 estimate may exceed a selector's existing 5000-character cap. Use the
snapshot's `.lengths.csv` to detect saturated/no-op configurations before
choosing a meaningful budget range. Do not change sentence ranking or retrieval
depth to make the controller look better.

## Available figures and report plan

[Overhead PNG](measurements/local_cpu_20261111/figures/overhead_vs_candidates.png)
and [PDF](measurements/local_cpu_20261111/figures/overhead_vs_candidates.pdf)
are generated from the measured synthetic timings.

| Requested figure | Status |
|---|---|
| Answer quality vs configured budget | Pending actual judge scores |
| Prompt tokens vs generation latency | Pending actual generation/usage |
| Answer quality vs average used tokens | Pending actual judge scores |
| Fixed vs adaptive allocation | Pending matched-cost quality results |
| Token reduction vs answer-quality reduction | Pending actual controls and judge scores |
| Controller overhead vs candidate evidence count | PNG and PDF available |

The [plot manifest](measurements/local_cpu_20261111/figures/plot_manifest.json)
records missing measurements; blank or fabricated quality curves are not
generated.

For the six-page report, use the same-selector control table, a matched-cost
fixed/adaptive figure, the quality/token frontier and a compact overhead panel.
Report sample sizes, evaluated-query coverage and paired intervals. Keep API
latency and local CPU cost separate, and label all estimated versus API tokens.

## Reproduction and remaining evaluations

The [README](README.md) gives complete asset installation prerequisites,
configuration names, API-free ablations, five-query real pilots, the unchanged
answer evaluator, practice/dev expansion and plotting commands.

Quick local reproduction:

```bash
cd /Users/burphh/Desktop/CS/MLS-CW/rag-adaptive-context-budget
export PYTHONPATH="$PWD/rag"
/opt/miniconda3/envs/diss/bin/python -m unittest discover -s rag/tests -v

/opt/miniconda3/envs/diss/bin/python -m extensions.adaptive_context_budget overhead \
  --output rag/results/budget_overhead_reproduction --counts 1 5 20 100 500 1000 --repeats 30
/opt/miniconda3/envs/diss/bin/python -m extensions.adaptive_context_budget plot \
  --overhead-csv rag/results/budget_overhead_reproduction/overhead.csv \
  --output rag/results/budget_overhead_reproduction/figures
```

For real evaluation: install LFS assets and the RAG dependencies, create/reuse
the practice ASR cache, configure ELM locally, prepare the frozen snapshot, then
run the explicitly selected five-query pilot and supplied evaluator from the
README. Broader generation requires `--allow-large-api-run` and an explicit
condition list; no full sweep was automatically triggered.

## Limitations and negative results

- Default budgets constrain an estimator, not true server tokenisation. A
  local matching generation tokenizer improves the text count; chat-template
  and server overhead still require separate accounting.
- Retrieval gaps are not calibrated answer confidence and can misallocate
  evidence. Superiority of adaptive allocation is currently unestablished.
- Exact-text deduplication is not semantic redundancy removal. Original
  no-budget controls are preserved; logging identifies duplicate discards.
- Sentence-level selectors already return whole sentences. Safe prefix packing
  may offer little benefit for these units, unlike multi-sentence documents.
- Prefix boundaries are punctuation-based and can misidentify abbreviations;
  they do not provide language-aware segmentation.
- Document qrels cannot determine whether a retained sentence contains an answer.
- Discrete packing may prevent exact average token-cost matching. Calibrating
  cost on one snapshot is not independent dev validation.
- Cached replay cannot measure fresh ASR end-to-end latency. Reconstructed
  downstream timing is labelled separately from measured wall time.
- No claim of preserved real answer quality or improved generation latency is
  justified until the missing course/API experiments are completed.
