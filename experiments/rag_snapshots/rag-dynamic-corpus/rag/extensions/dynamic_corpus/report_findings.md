# Dynamic corpus: measured findings

## Research question and hypothesis

Can a base/delta/tombstone index reduce maintenance cost while preserving exact
retrieval? The hypothesis was that incremental mutation avoids full matrix
reconstruction, while accumulated partitions increase query cost. Compaction might
balance those costs. The measurements below support that trade-off for these
**synthetic CPU fixtures only**; no course retrieval-quality or ELM answer-quality
result is claimed.

## Implementation and experimental controls

The extension maintains immutable FP32 base/delta arrays and copy-on-write
metadata, with one atomic state publication per batch. Updates invalidate the
previous row and append its replacement; deletes invalidate/remove active
mappings. Searches merge exact local top-k candidates into a global result.
Compaction materialises current active IDs, texts and vectors into a fresh base.
Both explicit and threshold-triggered compaction are implemented. The control
rebuilds once after every atomic batch, including all validation/copy costs.

The only shared source change extracts scoring/ranking helpers and adds in-memory
construction to the existing exact retriever. Controlled conditions all use the
same FP32 row-wise dot product and lexicographic ID policy for exact score ties.
Ordinary static retrieval retains its original BLAS/partition defaults. This
choice prevents storage-partition-dependent summation roundoff from changing
ties, and is checked against original static scoring within tolerance. It also
means these timings are not a benchmark of default BLAS against a different
scorer: all three maintenance strategies use the identical scorer.

The dynamic RAG adapter delegates to the original pipeline. Reranking, text
selection and optional context budgeting consume the same immutable retrieval
snapshot. The original ASR, encoder, generator, TF-IDF/MMR, reranker and evaluator
algorithms were not replaced. A completed write is visible to later retrieval;
in-flight requests retain their explicitly versioned old snapshot. Atomic batch
rollback and concurrent readers are tested. Persistence is not implemented.

Each measured condition starts from identical IDs/text/vector pairs, replays the
same seed-2026 mutation log, and uses fixed k=10 and four cycling probe vectors.
There is no K sweep. Initial active fraction is 80% of the source subset. Mixed
workloads contain exactly 40% insertions, 30% updates and 30% deletions. Refresh
updates retain original text/vector content. Correctness tests separately change
both embeddings and text, including repeated replacements and tie cases.

## Environment, sample counts and measurement definitions

- Apple M4, Mac16,1, 10 CPU cores (4 performance + 6 efficiency), 16 GB RAM;
  macOS 26.5.1 arm64, Python 3.11.11. Exact NumPy/Matplotlib versions, environment
  flags and source hashes are retained in the measurement metadata.
- No GPU used. BLAS/OpenMP thread limits were set to one; the controlled row-wise
  scorer is not BLAS GEMV. No CPU affinity/frequency or OS background-load control.
- Synthetic normalised Gaussian FP32 vectors, dimension **64**, source catalog
  24,000 rows. This is not the coursework enc_a embedding matrix.
- 21 factorised cases × 3 strategies × 3 repetitions = **189 workload runs**;
  **40,770 mutations**, **50,490 scheduled queries** in total. Diagnostic probes,
  warmups and separately traced memory replays are additional work.
- Each condition warms the same probes. Strategy order rotates across repeats.
  Means and sample standard deviations are in `summary.csv`; p95 is calculated
  per repetition, and the table averages those three p95 values.
- Maintenance is measured mutation wall time, including staging and any rebuild
  or automatic compaction. Those components are also reported separately.
  Workload time is maintenance + scheduled-query wall time. Initial construction,
  validation, logging and final diagnostic manual compaction are excluded; this
  is index service time, not live ASR-to-answer latency.
- Matrix/array storage is measured from current-state NumPy `nbytes`. UTF-8 text
  payload and shallow Python metadata are separately recorded. Additional traced
  peak allocation is measured by an untimed first-repeat replay with tracemalloc,
  excluding initial input buffers; it is not total process memory or RSS.

## Principal comparison

Source subset=2,000, initially active=1,600, 200 mutations, batch size one,
one scheduled query per mutation, automatic threshold=5% of physical delta rows
or base tombstones. Means across three repetitions:

| Metric | Full rebuild | Incremental | Incremental + compaction |
|---|---:|---:|---:|
| Insert wall latency (ms) | 0.8985 | 0.0806 | 0.0728 |
| Update wall latency (ms) | 0.9005 | 0.0786 | 0.0898 |
| Delete wall latency (ms) | 0.9007 | 0.0629 | 0.0588 |
| Query mean (ms) | 0.0397 | 0.4162 | 0.2383 |
| Query median (ms) | 0.0381 | 0.4112 | 0.2400 |
| Mean per-repeat p95 query (ms) | 0.0439 | 0.7299 | 0.4282 |
| Embedding storage before final manual compaction (MiB) | 0.3955 | 0.4248 | 0.4065 |
| Additional traced peak allocation, first repeat (MiB) | 1.9010 | 0.2410 | 1.4471 |
| Total mutation + automatic maintenance (ms) | 179.9571 | 14.9362 | 14.7425 |
| Scheduled query total (ms) | 7.9318 | 83.2344 | 47.6567 |
| Measured index workload sum (ms) | 187.8889 | 98.1706 | 62.3992 |
| Maintenance speedup vs rebuilding | 1.00× | 12.05× | 12.21× |
| Workload speedup vs rebuilding | 1.00× | 1.91× | 3.01× |
| Exact ordered top-k agreement | 100% | 100% | 100% |
| Retrieval/current-text correctness | PASS | PASS | PASS |

Total maintenance SDs were 2.4835 / 0.6892 / 0.2388 ms respectively. Query mean
SDs were 0.0023 / 0.0210 / 0.0028 ms. Incremental query mean was 10.49× the
rebuild query mean; with compaction it was 6.01×. These query penalties narrow the
maintenance advantage substantially. The small maintenance-time difference
between the two incremental conditions is not evidence that compaction is free;
the compacting condition performed one automatic compaction, costing 0.9318 ms.

Full rebuild spent 165.6873 ms reconstructing its matrix (staging 12.3412 ms).
Incremental performed no rebuild/compaction during mutations. Its 140 physical
delta rows and 110 invalid base rows remained allocated before final diagnostic
compaction. The compacting condition ended with 59 delta rows and 44 invalid base
rows. The raw receipt timings include the smaller remaining lock/publication and
caller overhead, so stage + rebuild/compaction need not equal wall time exactly.

## Ablations

### Batch size: control amortisation matters

All rows below replay the same 500-mutation mixed sequence at source N=2,000.
Values are mean maintenance ms/op; batching changes visibility/query scheduling
to after each whole batch for every strategy.

| Atomic batch | Full rebuild | Incremental | Incremental + 5% compaction | Incremental maintenance speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.9148 | 0.0841 | 0.0773 | 10.88× |
| 10 | 0.1000 | 0.0139 | 0.0216 | 7.18× |
| 100 | 0.0167 | 0.0072 | 0.0108 | 2.32× |
| 500 | 0.0084 | 0.0065 | 0.0084 | 1.30× |

Incremental maintenance was cheaper at every sampled batch size. Its advantage
shrunk as the full rebuild was amortised over more mutations. For batch 500 the
compacting strategy rebuilt after its one large batch, with practically the same
workload time as the control (19.12 vs 19.04 ms). No universal crossover or
significant winner can be inferred from that small difference.

Without compaction, the 500-operation workload was **slower overall** than full
rebuilding at all four batch sizes: e.g. batch one 512.68 vs 477.42 ms, batch ten
108.28 vs 66.96 ms. Lower update latency alone is insufficient.

### Mutation type

For 100 individual mutations, incremental maintenance speedups were 12.59× for
insertions, 12.23× for refresh updates, 16.70× for deletions and 13.94× for mixed
operations. Delete-only incremental mean query time stayed near the control
(0.0368 vs 0.0389 ms), because deletions create no additional vector chunks.
Insert/update means rose to 0.3237/0.3280 ms. This supports the interpretation that
partition handling, not just tombstones, drives much of the query overhead here.
The full per-operation and staging samples remain available; updates and inserts
still pay O(N) metadata copying in this implementation.

### Accumulated updates and manual compaction

After 300 mixed mutations in 30 batches of ten, uncompacted incremental probe
latency rose from 0.0312 to 0.2409 ms. It retained 30 delta chunks / 210 physical
delta rows and invalidated 10.375% of base rows. Explicit compaction reduced the
probe mean to 0.0314 ms and current embedding storage from 463,360 to 417,280
bytes, without changing retrieval. This is storage reclamation/partition removal,
not an accuracy improvement. The final probe uses only four queries per repeat;
scheduled means provide a larger sample.

### Automatic-compaction threshold

Same 300-operation, batch-ten workload; means over three repetitions:

| Threshold | Automatic compactions | Maintenance (ms) | Mean query (ms) | Workload sum (ms) |
|---|---:|---:|---:|---:|
| None | 0 | 4.16–4.64 | 0.1406–0.1438 | 46.35–47.79 |
| 1% | 10 | 13.2762 | 0.0414 | 25.6858 |
| 5% | 2 | 5.9703 | 0.0707 | 27.1906 |
| 10% | 1 | 5.5083 | 0.1055 | 37.1497 |
| 20% | 0 | 4.2391 | 0.1432 | 47.2054 |

The no-compaction ranges are repeated matched controls in the four threshold
cases, not four different configurations. At 20% the trigger was never reached.
Among sampled thresholds, **1% had the lowest workload sum**, and 5% reduced
maintenance further while accepting slower queries. The data do not establish
that an intermediate threshold always wins. Library default remains no automatic
compaction; the experiment's configurable default is 5%. Retune on real enc_a
dimensions and actual query/update frequency rather than adopting a universal
threshold from these fixtures.

### Queries per mutation: negative read-heavy finding

These short trials share ten individual mixed mutations; the threshold is not
reached, so the two incremental variants are algorithmically equivalent here.

| Queries per mutation | Full workload (ms) | Incremental (ms) | Incremental + compaction policy (ms) |
|---:|---:|---:|---:|
| 1 | 9.89 | 1.32 | 1.28 |
| 10 | 12.74 | 7.02 | 6.21 |
| 100 | 40.12 | 56.67 | 57.83 |

At 100 queries per mutation, rebuilding was faster overall despite its much more
expensive updates. The crossover is between the sampled 10 and 100 queries/op
for this log, not a measured exact threshold. Query overhead depends on accumulated
chunks, so extrapolating a single constant crossover to longer logs is unsafe.

### Corpus size

At 100 mixed mutations in batches of ten, no-compaction maintenance speedups rose
from 7.68× / 11.09× / 14.38× / 16.33× for source subsets 2k / 5k / 10k / 24k.
At source 24k, initially active **19,200**, rebuild maintenance was 123.809 ms,
incremental 7.582 ms; workload sums were 144.81 and 32.43 ms (4.46×). Vector
dimension remained 64, so the initial active matrix was only 4,915,200 bytes.

This small, cache-friendly corpus makes absolute rebuild costs modest and query
partition overhead prominent. Real coursework embeddings may have very different
dimensionality and cache behaviour. The 24k synthetic subset is a scaling fixture,
not the actual 24k-course-corpus experiment. Larger N increased observed relative
maintenance savings, but even this benchmark retains O(N) Python metadata copies.

## Correctness, quality and test evidence

Across all strategies, repeats and checkpoints: **4,860 query/checkpoint checks,
100% ordered top-k agreement, maximum score difference 0, zero logical-state,
inactive-ID or stale-text failures**. The oracle reconstructs its logical map
independently from the source mutation log, not from the index's own export.
Logical IDs/text/vector arrays are compared in addition to query rankings.

The unit/integration suite contains 41 tests. In the isolated branch, 40 pass and
one optional real budget-component check skips because that extension is not
merged. With the sibling adaptive-budget worktree on `PYTHONPATH`, **41 pass,
zero skips**. Tests cover invalid vectors/IDs, duplicates/missing records,
rollback, old snapshots, concurrent queries, compaction, deterministic replay,
base/delta merging, real TF-IDF/MMR, the existing reranker with a mocked model,
budget interface/real budget wrapper and the existing candidate validator. API
responses in integration tests are mocks and are not answer-quality evidence.

Course assets were absent: corpus JSONL, enc_a vectors/ID mapping, audio manifest,
model weights and real transcript/query cache. The course command exited 2 before
running; its output is saved as `measurements/course_attempt.txt`. OpenAI and
sentence-transformers were unavailable in the chosen CPU environment, and no ELM
key was available. Consequently **course Recall@10/MRR@10/NDCG@10/relevance
coverage and real generated/judged answer scores were not measured**.

The implemented course workflow reuses supplied metric functions, separately
reports original- and active-qrels scopes plus document availability, and suppresses
original-qrels interpretation after semantic replacements. The small RAG trial
reuses cached real ASR/query embeddings and the existing ELM generator/evaluator.
It shares one answer for identical prompts, explicitly labels cache hits, and
reports downstream wall time rather than fresh audio-to-answer latency. Its code
and schema were exercised with fixtures only. See the README for exact commands.

## Figures and report material

All nine figures have PNG and PDF versions under
`measurements/local_cpu_20261111/figures/`, generated solely from saved measured
CSVs. `plot_manifest.json` records all nine generated and none skipped.

| Figure basename | Suggested interpretation |
|---|---|
| `update_latency_vs_batch_size` | Rebuild amortisation erodes incremental advantage |
| `maintenance_time_comparison` | Principal update/maintenance advantage |
| `query_latency_vs_accumulated_updates` | Extra partitions raise query overhead |
| `query_latency_vs_compaction_threshold` | More frequent maintenance reduces query cost |
| `memory_vs_accumulated_updates` | Current matrix growth and reclamation, excludes RSS |
| `workload_time_vs_query_ratio` | Read-heavy workload reverses the winner |
| `maintenance_vs_corpus_size` | Relative maintenance savings increase with source N |
| `exact_topk_agreement` | No ranking defect at checked states |
| `cumulative_maintenance_cost` | How maintenance savings accumulate over time |

For the six-page report, prioritise the compact principal table, cumulative-cost
plot, read-heavy reversal and threshold plot, plus a short architecture paragraph
and exactness result. Label the data synthetic/CPU in every caption. Replace or
supplement these figures with enc_a results before making course-performance
claims.

## Threats to validity and next measurements

Three repeats and a shared laptop cannot establish small differences as
significant. Four corpus-vector probes are not a representative audio-query
distribution. Warm caches, fixed 64-dimensional vectors and small active matrices
favour different costs from real enc_a. The simple full-rebuild control copies and
validates all vectors, including staging before reconstruction; a more specialised
control could be faster. Batch queries run after publication, so batching also
changes which intermediate states are exposed. Snapshot-held memory, process RSS,
concurrent latency stalls, crash recovery and real model/API variability are not
measured. The factorised suite is not a full factorial search or proof of an
optimal threshold. Reproduce on immutable enc_a assets, prepare five real queries,
and run the opt-in matched RAG trial before drawing answer-quality conclusions.
