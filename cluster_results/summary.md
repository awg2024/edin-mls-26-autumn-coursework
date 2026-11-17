# Consolidated experiment results and report rationale

Prepared on 16 November 2026 from the downloaded `elm-20261115-02` run and the existing archived CUDA experiments. The run ID is retained even though the continuation ran on 16 November. This document consolidates the available results; raw evidence remains at the linked paths. No experiments, tests or ELM requests were rerun to prepare it.

## 1. Findings to lead with

- **Dynamic corpus:** incremental maintenance with automatic compaction took **31.883 ms** for 200 mutations versus **479.409 ms** for rebuilding: a **15.04× maintenance speedup**. Including the 200 scheduled reads, workload time fell from **515.636 to 137.278 ms**, a **3.76× speedup**, in the principal 2,000-source-record experiment. All **6,075 checkpoint/query exactness checks** passed. Search alone was slower: **0.527 versus 0.181 ms/query**. This is an update/read tradeoff, not a universal retrieval speedup.
- **Adaptive context:** relative to unbudgeted MMR, mean API input tokens fell from **1,250.2 to 980.0** (**21.61%**) and estimated evidence tokens from **921.8 to 700.6** (**24.00%**). Mean answer score fell from **0.4167 to 0.2333**. A fixed budget matched to the adaptive mean evidence cost scored **0.3167**. This pilot does **not** demonstrate an adaptive quality advantage.
- **A crucial saturation control:** `mmr_fixed_2048_skip` and `mmr_none` sent **identical prompt strings and contexts for all five queries**, yet their scores were **0.5500 and 0.4167**. All five answer strings differed. Independent generation/judging variation prevents attributing that score increase to a budget that changed no prompt.
- **ASR:** all **33 supplied kernel cases** and both full-model checks passed. CUDA-all transcription was **3.91× slower** than the PyTorch reference (**1,638.2 versus 419.0 ms**) with slightly lower peak allocated GPU memory. Correctness was achieved; model-level acceleration was not.
- **Coverage:** five matched practice queries were generated and judged in each of ten budget arms and four dynamic conditions. These are **five distinct questions**, not 50 or 70 independent questions, and not the teammate's 350-query K sweep.

## 2. Research questions and why these experiments are justified

### Recommended overarching research question

> How do incremental corpus maintenance and retrieval-aware evidence budgeting change the systems cost, retrieval correctness and answer quality of an audio-query RAG pipeline?

Suggested title: **Cost and quality tradeoffs in audio-query RAG: dynamic corpus maintenance and adaptive evidence budgets**.

The two mechanisms were evaluated separately. There is no factorial experiment combining dynamic updates and adaptive budgets, so do not claim a measured speedup or quality gain for a combined system.

### RQ1 — dynamic corpus maintenance

> Under what update batch sizes, query-to-update ratios and corpus sizes does exact incremental retrieval with compaction reduce maintenance and total workload cost relative to rebuilding, while returning the same results from the same active corpus?

**Justification:** a document collection changes through insertions, updates and deletions. Reconstructing its complete index after every committed batch is a simple correctness-preserving control, but repeatedly copies the active matrix. A base-plus-delta design can defer that work; accumulated chunks and tombstones can make reads more expensive. The experiment is justified by measuring both sides of that tradeoff and checking exactness rather than assuming cheaper writes are an overall improvement.

The controls answer distinct questions: rebuild versus incremental isolates maintenance strategy; incremental versus incremental-with-compaction isolates whether maintenance recovers enough search efficiency to justify its own cost; query-ratio and batch-size ablations establish when each choice pays off. The results support a workload-dependent answer, including a case where rebuilding wins.

### RQ2 — adaptive evidence allocation

> At a fixed retrieval depth and sentence selector, can a normalised retrieval-score-gap rule reduce ELM input tokens while preserving judged answer quality, and does it outperform a fixed budget at matched mean evidence-token cost?

**Justification:** passing less evidence can reduce prompt size but can also discard the sentences needed to answer a question. A fixed budget is a simple alternative to the proposed adaptive rule. Comparing only against unbudgeted selected evidence cannot establish whether adaptation helps; matching average used tokens asks whether assigning the budget differently between questions produces better quality for a similar input cost. The existing selector's character cap still applies to unbudgeted evidence.

The experiment is still useful when the proposed rule loses. It identifies a limitation of this score-gap heuristic, shows the difference between retaining a relevant document and retaining its useful sentences, and exposes generation variability through the saturation control. It does not establish that every adaptive policy fails.

### Supporting ASR question

> How do the custom CUDA kernels compare with the supplied numerical references and PyTorch ASR in correctness, latency and allocated memory, and which kernel design choices explain their performance?

Treat ASR validation/kernel ablations as a supporting systems study if the report covers the full coursework. Keep its GPU timing claims separate from CPU retrieval and network-dependent ELM timings.

### Exact baselines and what each comparison establishes

| Experiment / treatment | Control | What the comparison establishes |
| --- | --- | --- |
| Full ASR with CUDA-all | Same clip/model with `ASR_CUDA=none` (PyTorch) | Integration correctness and measured model latency/memory cost |
| Individual CUDA kernels | Supplied CPU/cuBLAS references; naive GEMM for throughput | Numerical correctness and shape-specific efficiency |
| Incremental corpus | Full rebuild after each atomic batch, same active records/vectors | Cost of maintenance while preserving exact retrieval |
| Incremental + automatic compaction | Incremental without automatic compaction | Cost/benefit of cleaning delta rows and tombstones |
| Dynamic-after RAG | Freshly rebuilt identical after-state | Evidence and retrieval consistency at identical corpus state |
| Dynamic-before/after versus original static | Full original 24,000-record static corpus | Effect of a different available corpus; not an isolated indexing-quality comparison |
| TF-IDF + fixed 512 | `tfidf_none` | Budgeting effect within the existing TF-IDF selector |
| MMR fixed/adaptive | `mmr_none` | Budgeting effect within the existing MMR selector |
| MMR adaptive | `mmr_matched_skip` (fixed 789) | Adaptive allocation versus a similar mean used-token cost |
| Any selector/budget versus Simple | `simple_none` | Practical pipeline comparison that includes a selector change |
| MMR fixed 2048 | `mmr_none` | Saturation/repeat-generation control because the prompts are identical here |

The teammate's 350-query K sweep supplies context for the project, but is not a matched baseline for these five-query experiments. Do not attribute differences between those datasets/runs to our extensions.

## 3. Evidence, completion and controlled settings

| Evidence group | Available result | Scope |
| --- | --- | --- |
| Supplied ASR validation | 33/33 cases; both model modes PASS | Imported original cluster logs; not rerun in this RAG continuation |
| RAG tests | 57 adaptive-budget and 41 dynamic tests PASS | Remote generation mocked in tests; separate from real ELM evidence |
| Dynamic course suite | 21 cases × 3 strategies × 3 repeats = 189 repeat summaries | Actual course enc_a vectors, five cached real ASR/query vectors |
| Budget ELM pilot | 10 arms × 5 successful answers and judgements | Real ELM; fixed retrieval k=5; no reranker |
| Dynamic ELM trial | 4 conditions × 5 scored rows | Real ELM plus explicit identical-prompt answer reuse |
| Separate budget packing/range/overhead suites | Not present in downloaded run | No claim that planned measurements completed |
| Budget plots | Five PNG/PDF figure pairs available | No candidate-count overhead figure |
| Dynamic plots | No PNG/PDF files present; figures directory contains only cache data | Raw measurements and summary tables are available |

Sources: [recorded commands](elm-20261115-02/commands.jsonl), [source hashes](elm-20261115-02/source_sha256.json), [dynamic suite manifest](elm-20261115-02/dynamic_course/experiment_manifest.json), [budget manifest](elm-20261115-02/budget_pilot/run_manifest.json), and [ASR import provenance](elm-20261115-02/manual_asr_validation.json).

The original historical failures remain recorded: missing corpus inputs in the first preparation attempt, missing ELM key, and an existing empty generation log. Later query preparation, generation and judging succeeded. The incomplete automatic `prepare` entry is historical; the saved inputs were subsequently prepared manually. The command ledger does not contain every manually invoked preparation/benchmark command, so use their saved manifests, logs and measurements too.

Controlled RAG settings:

- First five practice audio queries, shared real ASR transcript cache and local enc_a query encoder. ASR used the CUDA-all worker; CPU encoding was retained. The preparation log records a CPU float16 encoder warning; no dtype change was made in this continuation.
- Corpus source: **24,000 records, 384-dimensional enc_a vectors**. Dynamic index computations are FP32 CPU operations, not GPU search benchmarks.
- Budget retrieval **k=5**, existing Simple/TF-IDF/MMR builders, MMR lambda **0.7**, selector character cap **5,000**, no reranker. Dynamic systems search uses **k=10**. The dynamic trial config records `final_k=5`, but without a reranker the pipeline forwards the top-10 retrieval to the builder; it does not truncate to five automatically. Thus do not treat its Simple arm as identical to the budget pilot's k=5 baseline.
- Generator: **utter-project/EuroLLM-22B-Instruct-2512**, temperature **0.1**. All 50 budget response records identify that model. Judge: **utter-project/EuroLLM-22B-Instruct-2512**, one judgement repeat, supplied rubric `rag-answer-rubric-v0.1` and script `v0.3-public-answer-judge`. The judge uses temperature 0 and a shared cache.
- The judge's question is the **ASR transcript**, and its gold evidence is reconstructed from original public qrels. Correctness, grounding and completeness are each 0–4; the normalised score is their sum divided by 12. It is a rubric score, not accuracy, WER or a percentage of correct answers.
- Public split coverage is **5/350 = 1.43%**; within the planned five-query pilot, generated/judged coverage is **100%**.
- Subsequent RAG timings reuse ASR/encoder inputs. These are not fresh audio-to-answer wall times. Budget limits estimate evidence tokens as ceil(UTF-8 bytes/4); actual API input/output usage is reported separately. No currency cost was recorded.

## 4. Dynamic corpus: systems measurements

### Principal comparison and interpretation

The principal case selects a 2,000-record source subset, starts with 1,600 active documents, performs 200 mixed operations (80 inserts, 60 refresh updates, 60 deletes) in batches of one, and schedules one read per operation. The final active count is 1,620. Automatic compaction uses a 5% threshold. Each strategy sees the same ordered mutation log, query vectors and exact tie/scoring policy. Strategy order rotates over three repeats, with equal warmup.

An update in this dataset is a refresh using an original compatible text/vector pair, not a new semantic edit or fresh document embedding. The experiment measures index maintenance, not an embedding service or the time to encode newly written documents.

| Strategy | Mutation total ms, mean ± SD | Query ms, mean ± SD | Mean per-repeat p95 ms | Workload ms, mean ± SD | Embedding MiB | Mutation speedup | Workload speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_rebuild | 479.409 ± 24.900 | 0.1811 ± 0.0117 | 0.2203 | 515.636 ± 27.215 | 2.373 | 1.00× | 1.00× |
| incremental | 31.280 ± 1.123 | 0.8884 ± 0.0363 | 1.4943 | 208.959 ± 8.303 | 2.549 | 15.33× | 2.47× |
| incremental_compact | 31.883 ± 0.223 | 0.5270 ± 0.0011 | 0.8834 | 137.278 ± 0.315 | 2.439 | 15.04× | 3.76× |

Mutation time includes batch application and automatic maintenance. Workload time is the sum of mutation wall time and scheduled query wall time; it excludes initial construction, oracle checks, serialisation, the separate traced-memory replay and final diagnostic manual compaction. “Mean per-repeat p95” is not a pooled percentile. Storage is current embedding matrices only, not total RSS or retained reader snapshots.

The compacting strategy cuts principal maintenance time by **93.35%**, but its mean search latency is **2.91×** the rebuilt strategy. Compared with incremental without compaction, it cuts mean query latency by **40.68%** and workload time by **34.30%** for only a small increase in mutation time. One automatic compaction occurs per principal repeat.

The query penalty is consistent with scoring/ranking multiple delta chunks in Python and merging candidates, while rebuilding keeps one compact matrix. These mechanisms follow from the implementation; no hardware-counter profile isolates their individual contributions. Metadata copies are O(active IDs), so incremental maintenance is not O(1).

After the final diagnostic manual compaction, mean query time is **0.1405 ms** for incremental and **0.1386 ms** for automatic-compaction mode, close to **0.1384 ms** for rebuild. Manual compaction costs **3.257**, **2.605** and **2.405 ms**, respectively; these diagnostic costs are outside the headline workload totals.

### Ablations: all 21 case configurations

The following tables include every configuration. Each cell is a mean across three repeats. Source N differs from active N (initially 80% of source N). `I` = incremental, `C` = incremental with automatic compaction. Workload speedup is rebuild workload / treatment workload; values below 1 favour rebuilding. Source CSVs retain SDs, mutation-type costs, initial build time, staging/rebuild/compaction breakdowns and memory statistics.

#### Principal

| Case parameters | Rebuild mutation ms | I mutation ms | C mutation ms | Query ms: rebuild / I / C | Workload ms: rebuild / I / C | Workload speedup I / C |
| --- | --- | --- | --- | --- | --- | --- |
| N=2000; ops=200; batch=1; mixed; reads/op=1; threshold=5% | 479.409 | 31.280 | 31.883 | 0.1811 / 0.8884 / 0.5270 | 515.636 / 208.959 / 137.278 | 2.47× / 3.76× |

#### Batch size

| Case parameters | Rebuild mutation ms | I mutation ms | C mutation ms | Query ms: rebuild / I / C | Workload ms: rebuild / I / C | Workload speedup I / C |
| --- | --- | --- | --- | --- | --- | --- |
| N=2000; ops=500; batch=1; mixed; reads/op=1; threshold=5% | 1198.029 | 96.248 | 88.260 | 0.1769 / 1.9046 / 0.5678 | 1286.493 / 1048.566 / 372.147 | 1.23× / 3.46× |
| N=2000; ops=500; batch=10; mixed; reads/op=1; threshold=5% | 147.947 | 17.777 | 28.687 | 0.1496 / 0.4668 / 0.2105 | 222.742 / 251.156 / 133.944 | 0.89× / 1.66× |
| N=2000; ops=500; batch=100; mixed; reads/op=1; threshold=5% | 24.400 | 7.582 | 13.195 | 0.1451 / 0.1865 / 0.1404 | 96.961 / 100.844 / 83.409 | 0.96× / 1.16× |
| N=2000; ops=500; batch=500; mixed; reads/op=1; threshold=5% | 14.884 | 7.074 | 8.768 | 0.1428 / 0.1767 / 0.1380 | 86.271 / 95.434 / 77.787 | 0.90× / 1.11× |

#### Mutation type

| Case parameters | Rebuild mutation ms | I mutation ms | C mutation ms | Query ms: rebuild / I / C | Workload ms: rebuild / I / C | Workload speedup I / C |
| --- | --- | --- | --- | --- | --- | --- |
| N=2000; ops=100; batch=1; insert; reads/op=1; threshold=5% | 249.991 | 15.834 | 18.998 | 0.1803 / 0.6785 / 0.5400 | 268.024 / 83.688 / 72.999 | 3.20× / 3.67× |
| N=2000; ops=100; batch=1; update; reads/op=1; threshold=5% | 242.898 | 16.669 | 19.027 | 0.1788 / 0.7212 / 0.5066 | 260.774 / 88.788 / 69.688 | 2.94× / 3.74× |
| N=2000; ops=100; batch=1; delete; reads/op=1; threshold=5% | 234.848 | 9.636 | 13.145 | 0.1763 / 0.1433 / 0.1425 | 252.483 / 23.965 / 27.400 | 10.54× / 9.21× |
| N=2000; ops=100; batch=1; mixed; reads/op=1; threshold=5% | 254.829 | 14.288 | 14.313 | 0.1860 / 0.5045 / 0.5049 | 273.426 / 64.734 / 64.801 | 4.22× / 4.22× |

#### Accumulation

| Case parameters | Rebuild mutation ms | I mutation ms | C mutation ms | Query ms: rebuild / I / C | Workload ms: rebuild / I / C | Workload speedup I / C |
| --- | --- | --- | --- | --- | --- | --- |
| N=2000; ops=300; batch=10; mixed; reads/op=1; threshold=5% | 86.074 | 9.440 | 15.422 | 0.1436 / 0.3446 / 0.2193 | 129.144 / 112.821 / 81.215 | 1.14× / 1.59× |

#### Threshold

| Case parameters | Rebuild mutation ms | I mutation ms | C mutation ms | Query ms: rebuild / I / C | Workload ms: rebuild / I / C | Workload speedup I / C |
| --- | --- | --- | --- | --- | --- | --- |
| N=2000; ops=300; batch=10; mixed; reads/op=1; threshold=1% | 95.845 | 12.425 | 38.809 | 0.1537 / 0.3405 / 0.1510 | 141.968 / 114.577 / 84.096 | 1.24× / 1.69× |
| N=2000; ops=300; batch=10; mixed; reads/op=1; threshold=5% | 89.071 | 10.129 | 15.908 | 0.1430 / 0.3426 / 0.2050 | 131.976 / 112.894 / 77.401 | 1.17× / 1.71× |
| N=2000; ops=300; batch=10; mixed; reads/op=1; threshold=10% | 90.509 | 9.823 | 13.678 | 0.1463 / 0.3387 / 0.2817 | 134.396 / 111.430 / 98.189 | 1.21× / 1.37× |
| N=2000; ops=300; batch=10; mixed; reads/op=1; threshold=20% | 95.072 | 9.834 | 10.205 | 0.1535 / 0.3402 / 0.3397 | 141.125 / 111.893 / 112.122 | 1.26× / 1.26× |

#### Query ratio

| Case parameters | Rebuild mutation ms | I mutation ms | C mutation ms | Query ms: rebuild / I / C | Workload ms: rebuild / I / C | Workload speedup I / C |
| --- | --- | --- | --- | --- | --- | --- |
| N=2000; ops=10; batch=1; mixed; reads/op=1; threshold=5% | 32.901 | 1.527 | 1.441 | 0.2243 / 0.2009 / 0.1980 | 35.144 / 3.537 / 3.421 | 9.94× / 10.27× |
| N=2000; ops=10; batch=1; mixed; reads/op=10; threshold=5% | 34.219 | 1.632 | 1.588 | 0.1699 / 0.1790 / 0.1801 | 51.207 / 19.533 / 19.600 | 2.62× / 2.61× |
| N=2000; ops=10; batch=1; mixed; reads/op=100; threshold=5% | 38.806 | 3.271 | 3.089 | 0.1331 / 0.1862 / 0.1804 | 171.857 / 189.484 / 183.465 | 0.91× / 0.94× |

#### Corpus size

| Case parameters | Rebuild mutation ms | I mutation ms | C mutation ms | Query ms: rebuild / I / C | Workload ms: rebuild / I / C | Workload speedup I / C |
| --- | --- | --- | --- | --- | --- | --- |
| N=2000; ops=100; batch=10; mixed; reads/op=1; threshold=5% | 34.705 | 3.107 | 3.050 | 0.1671 / 0.2111 / 0.2079 | 51.416 / 24.220 / 23.836 | 2.12× / 2.16× |
| N=5000; ops=100; batch=10; mixed; reads/op=1; threshold=5% | 98.188 | 5.263 | 5.573 | 0.3075 / 0.3312 / 0.3350 | 128.939 / 38.385 / 39.071 | 3.36× / 3.30× |
| N=10000; ops=100; batch=10; mixed; reads/op=1; threshold=5% | 203.055 | 10.082 | 10.270 | 0.6164 / 0.5509 / 0.5419 | 264.693 / 65.170 / 64.455 | 4.06× / 4.11× |
| N=24000; ops=100; batch=10; mixed; reads/op=1; threshold=5% | 499.722 | 30.457 | 28.684 | 2.7854 / 2.6070 / 2.0908 | 778.264 / 291.160 / 237.769 | 2.67× / 3.27× |

What these ablations justify:

1. **Batching is a real control.** At 500 operations, rebuild maintenance drops from **1,198.03 ms** at batch=1 to **14.88 ms** at batch=500 because it rebuilds once per batch. The compacting strategy's workload advantage shrinks from **3.46× to 1.11×**. A baseline forced to rebuild per document even for a bulk batch would overstate the benefit.
2. **Read-heavy workloads can favour rebuilding.** For the ten-operation workload, compacting incremental is **10.27×** faster overall at one read/operation and **2.61×** at ten reads/operation, but only **0.94×** at 100 reads/operation (**183.47 versus 171.86 ms**). No automatic compaction is triggered in these short cases. These points locate an observed tradeoff, not a universal break-even query ratio.
3. **Compaction frequency has a cost.** In the threshold sweep, 1%, 5%, 10% and 20% trigger **10, 2, 1 and 0** automatic compactions. Compacting workloads are **84.10, 77.40, 98.19 and 112.12 ms**. Five percent gives the smallest measured total among these four settings; 1% gives lower search time but higher maintenance. Do not call 5% globally optimal.
4. **Workload type matters.** Tombstone-only deletion is particularly cheap: incremental without compaction is **10.54×** faster overall than rebuild in the delete-only case, versus **9.21×** with automatic compaction. Compaction is not always worth paying for short workloads.
5. **Scale changes the balance.** At 24,000 source records (19,200 initially active), compacting maintenance is **28.68 versus 499.72 ms** and workload time **237.77 versus 778.26 ms**, a **3.27×** workload speedup. This is a different 100-operation/batch=10 experiment from the 2,000-record principal case.
6. **Accumulation is measurable.** In the 300-operation/batch=10 case, mean query latency is **0.3446 ms** without automatic compaction and **0.2193 ms** with it. State-history CSVs show delta/tombstone growth and cleanup. A single averaged query time should not be presented as a complete latency-over-time curve.

Sources: [all 63 case/strategy summaries](elm-20261115-02/dynamic_course/summary.csv), [all 189 repeat summaries](elm-20261115-02/dynamic_course/all_summary_per_repeat.csv), [Excel systems export](elm-20261115-02/excel/dynamic_systems.csv), and per-case `batch_timings.csv`, `operation_timings.csv`, `query_timings.csv` and `state_history.csv` under `dynamic_course/`.

### Exactness and retrieval-quality scope

The independent oracle rebuilds the same logical active state at checkpoints and checks ranked IDs, scores, current text and membership. Across **6,075 checks**, there are **zero top-k, score, logical-state or stale/inactive-text mismatches**, with maximum reported score error **0**. There are **666 qrels-evaluated checks**; these are repeated state/query observations, not 666 distinct questions.

Absolute qrels quality is a separate issue: most ablations use a small random source subset, which can omit the relevant documents altogether. In the principal final state, all five queries have **zero relevant-document availability**, so original-qrels Recall@10 is 0 and active-qrels metrics are undefined. This does not contradict exactness. It also means the principal case cannot establish that the index answers those questions well.

The final-state quality metrics below are shared by all three strategies at identical corpus state; N/A means no eligible active relevant documents. Original recall and availability average over all five queries; active metrics average over only the eligible queries. Each row is averaged across repeated checks of the same final state.

| Case | Active-eligible queries | Relevant availability | Original Recall@10 | Active Recall@10 | Active MRR@10 | Active NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| principal 200 ops | 0/5 | 0.000 | 0.000 | N/A | N/A | N/A |
| batch_size batch=1 | 0/5 | 0.000 | 0.000 | N/A | N/A | N/A |
| batch_size batch=10 | 0/5 | 0.000 | 0.000 | N/A | N/A | N/A |
| batch_size batch=100 | 0/5 | 0.000 | 0.000 | N/A | N/A | N/A |
| batch_size batch=500 | 0/5 | 0.000 | 0.000 | N/A | N/A | N/A |
| mutation_type insert | 1/5 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 |
| mutation_type update | 0/5 | 0.000 | 0.000 | N/A | N/A | N/A |
| mutation_type delete | 0/5 | 0.000 | 0.000 | N/A | N/A | N/A |
| mutation_type mixed | 1/5 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 |
| accumulation 300 ops | 1/5 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 |
| threshold 1% | 1/5 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 |
| threshold 5% | 1/5 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 |
| threshold 10% | 1/5 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 |
| threshold 20% | 1/5 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 |
| query_ratio reads/op=1 | 0/5 | 0.000 | 0.000 | N/A | N/A | N/A |
| query_ratio reads/op=10 | 0/5 | 0.000 | 0.000 | N/A | N/A | N/A |
| query_ratio reads/op=100 | 0/5 | 0.000 | 0.000 | N/A | N/A | N/A |
| corpus_size N=2000 | 1/5 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 |
| corpus_size N=5000 | 1/5 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 |
| corpus_size N=10000 | 1/5 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 |
| corpus_size N=24000 | 4/5 | 0.700 | 0.433 | 0.667 | 0.375 | 0.370 |

Sources: [exactness totals](elm-20261115-02/dynamic_course/correctness_summary.json) and [final-state quality/export counts](elm-20261115-02/excel/dynamic_retrieval.csv). Original-qrels quality is unsuitable for semantic replacements; this run uses refresh operations rather than replacements.

## 5. Dynamic corpus: real ELM integration trial

Conditions are: full original static corpus; initial dynamic corpus with 80% of records; dynamic corpus after 20 mixed refresh-mode mutations (8 inserts, 6 refresh updates, 6 deletes); and a freshly rebuilt identical after-state. The correct maintenance control is **dynamic-after versus rebuilt-after**. Original-static versus the smaller dynamic corpus also changes document availability and uses the ordinary static scoring path, so it cannot isolate indexing performance or quality.

| Condition | Rows / judged | Generation cache hits | Mean retrieval ms | Cached-input downstream wall ms | Uncached generation ms | Correctness / grounding / completeness (0–4) | Mean score (0–1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_static | 5 / 5 | 0 | 9.180 | 4559.376 | 4549.838 | 1.20 / 1.00 / 0.80 | 0.2500 |
| dynamic_before | 5 / 5 | 1 | 17.735 | 3411.728 | 4242.221 | 1.40 / 1.00 / 1.00 | 0.2833 |
| dynamic_after | 5 / 5 | 5 | 8.350 | 8.522 | N/A | 1.40 / 1.00 / 1.00 | 0.2833 |
| freshly_rebuilt_after | 5 / 5 | 5 | 7.738 | 7.912 | N/A | 1.40 / 1.00 / 1.00 | 0.2833 |

There are **9 distinct generated prompts and 11 cache-hit rows** across the 20 dynamic rows. All five dynamic-after versus rebuilt-after matching checks pass, covering IDs/scores, contexts, prompt, selected evidence and the reused answer.

Direct inspection also shows that the 20 mutations did **not change the retrieved IDs/scores or prompts for any of the five queries between dynamic-before and dynamic-after**. The trial therefore confirms integration and consistency for these queries, but does not demonstrate how answer quality changes when an edit affects retrieved evidence. The unchanged answer scores are expected under prompt caching.

Do not compare **8.52 ms** cached downstream time after updates with **4,559.38 ms** for original-static and call it a RAG speedup: after-state generations are cache hits. The pure index suite is the appropriate systems-cost comparison. Judge gold still comes from the original corpus, which limits interpretation after deletions.

Sources: [trial configuration](elm-20261115-02/dynamic_trial/config.json), [actual mutations](elm-20261115-02/dynamic_trial/mutations.jsonl), [five matching checks](elm-20261115-02/dynamic_trial/matched_checks.json), [full evidence/answer/timing/score table](elm-20261115-02/dynamic_trial/rag_comparison.csv) and [Excel aggregate](elm-20261115-02/excel/dynamic_answers.csv).

## 6. Adaptive context: all real ELM pilot conditions

The controller applies an ordered `skip` packing policy after the existing selector. At least three valid retrieval scores produce confidence `(largest − second)/(largest − smallest)`. Budgets vary from 256 to 2,048 using thresholds 0.1/0.6 and gamma=1; a larger normalised gap allocates fewer tokens. This is a retrieval-ambiguity heuristic, not a calibrated probability of answering correctly. Qrels and judge scores were not used by the controller.

The matched fixed budget is **789 estimated evidence tokens**. It was selected using lengths/scores on the same snapshot, without answer labels, to approximately match adaptive mean used tokens. It is a descriptive cost control, not a policy tuned on a separate training set.

All ten rows have **5/5 generated and 5/5 judged** queries. Times below are milliseconds; SDs are across the five questions, not independent repeated generations for each question.

| Condition | Budget (estimated tokens) | Used evidence tokens, mean | API input / output tokens, mean | Generation ms, mean ± SD | Controller ms, mean | Answer score, mean ± SD |
| --- | --- | --- | --- | --- | --- | --- |
| simple_none | off | 1209.2 | 1178.4 / 212.8 | 4098.2 ± 1302.4 | 0.000 | 0.2500 ± 0.2357 |
| tfidf_none | off | 921.8 | 1250.2 / 216.6 | 3688.2 ± 1844.4 | 0.000 | 0.2667 ± 0.2528 |
| mmr_none | off | 921.8 | 1250.2 / 266.0 | 4529.1 ± 1782.9 | 0.000 | 0.4167 ± 0.1021 |
| tfidf_fixed_512_skip | 512 | 480.6 | 711.0 / 172.2 | 3010.2 ± 1271.4 | 0.834 | 0.2500 ± 0.2282 |
| mmr_fixed_512_skip | 512 | 482.2 | 713.8 / 251.6 | 4635.9 ± 1206.7 | 0.837 | 0.3333 ± 0.4082 |
| mmr_adaptive_skip | adaptive; mean 1012.2 | 700.6 | 980.0 / 209.8 | 3570.3 ± 2464.3 | 0.966 | 0.2333 ± 0.2157 |
| mmr_matched_skip | 789 | 699.6 | 980.6 / 284.8 | 4936.6 ± 2970.2 | 1.001 | 0.3167 ± 0.2075 |
| mmr_fixed_256_skip | 256 | 244.0 | 395.6 / 146.6 | 2531.6 ± 1104.1 | 0.844 | 0.2667 ± 0.2595 |
| mmr_fixed_1024_skip | 1024 | 835.4 | 1140.8 / 320.4 | 5476.2 ± 2257.0 | 0.867 | 0.2167 ± 0.2007 |
| mmr_fixed_2048_skip | 2048 | 921.8 | 1250.2 / 261.0 | 4614.4 ± 2078.4 | 0.918 | 0.5500 ± 0.2541 |

Estimated evidence tokens exclude the question, instructions and chat wrapper. API input tokens cover the actual request using the service's tokenizer, so they need not equal or exceed the estimate. Input/output token totals are cost proxies; no monetary bill or fixed latency saving was measured.

### Same-selector cost and quality comparisons

Percentage reductions in this table are calculated as `100 × (1 − treatment mean / baseline mean)`. They are ratios of means, not the mean of per-query percentage reductions. A negative latency reduction means the treatment was slower.

| Treatment | Control | Evidence-token reduction | API-input reduction | Mean generation-latency reduction | Mean score change |
| --- | --- | --- | --- | --- | --- |
| tfidf_fixed_512_skip | tfidf_none | 47.86% | 43.13% | 18.38% | -0.0167 |
| mmr_fixed_512_skip | mmr_none | 47.69% | 42.91% | -2.36% | -0.0833 |
| mmr_adaptive_skip | mmr_none | 24.00% | 21.61% | 21.17% | -0.1833 |
| mmr_matched_skip | mmr_none | 24.11% | 21.56% | -9.00% | -0.1000 |
| mmr_fixed_256_skip | mmr_none | 73.53% | 68.36% | 44.10% | -0.1500 |
| mmr_fixed_1024_skip | mmr_none | 9.37% | 8.75% | -20.91% | -0.2000 |
| mmr_fixed_2048_skip | mmr_none | 0.00% | 0.00% | -1.88% | 0.1333 |

Main conclusions:

- **TF-IDF fixed 512** reduces API input tokens by **43.13%** with mean score **0.2500 versus 0.2667**. The point estimate suggests a comparatively small score change, but five queries cannot establish quality equivalence.
- **MMR fixed 512** reduces API input tokens by **42.91%**, while mean score falls from **0.4167 to 0.3333**. Generation is **2.36% slower** on average despite the shorter input. Input length alone does not determine network/model latency; output length and request variability also matter.
- **MMR adaptive** reduces API input tokens by **21.61%** and mean latency by **21.17%**, but retains only **56% of the baseline mean rubric score**. This is a worse observed quality/cost balance than the intended quality-preserving hypothesis.
- **Matched fixed versus adaptive** uses **699.6 versus 700.6** mean estimated evidence tokens, a **0.143%** matching error, with **980.6 versus 980.0** mean API input tokens. Adaptive score is lower by **0.0833**, and its faster observed generation cannot be assigned solely to allocation because output lengths/network variation differ. The matched-quality interval crosses zero.
- **MMR fixed 256** gives the largest input reduction (**68.36%**) with score **0.2667**. This is a measured pilot operating point, not a validated deployment choice.
- **MMR fixed 1024** scores lower and takes longer than unbudgeted MMR in this sample, despite saving **8.75%** of API input tokens. The observed curve is not monotonic.
- **MMR fixed 2048** removes no evidence and changes no prompt for any query. Its higher score is a repeat-generation observation, not an optimisation benefit. Do not select it as the “best budget” from this pilot.
- Comparing budgeted MMR directly with `simple_none` includes both sentence selection and budgeting. Attribute budget-only effects to the matched MMR control, not to the Simple control.

### Paired quality differences and uncertainty

Differences are treatment minus baseline in normalised score units. Intervals are the existing 2,000-resample paired-query percentile bootstrap (seed 2026). They condition on these five queries and these generated/judged answers; they do not include repeated-generation or repeated-judge uncertainty, and no multiple-comparison correction is recorded.

| Baseline | Treatment | Paired n | Mean score difference | Bootstrap 95% interval |
| --- | --- | --- | --- | --- |
| mmr_matched_skip | mmr_adaptive_skip | 5 | -0.0833 | [-0.2167, 0.0333] |
| mmr_none | mmr_adaptive_skip | 5 | -0.1833 | [-0.3000, -0.0667] |
| mmr_none | mmr_fixed_1024_skip | 5 | -0.2000 | [-0.3000, -0.1000] |
| mmr_none | mmr_fixed_2048_skip | 5 | 0.1333 | [-0.0333, 0.3667] |
| mmr_none | mmr_fixed_256_skip | 5 | -0.1500 | [-0.3000, 0.0167] |
| mmr_none | mmr_fixed_512_skip | 5 | -0.0833 | [-0.2833, 0.1667] |
| mmr_none | mmr_matched_skip | 5 | -0.1000 | [-0.2167, -0.0167] |
| tfidf_none | tfidf_fixed_512_skip | 5 | -0.0167 | [-0.1833, 0.2167] |

The adaptive-versus-unbudgeted interval is entirely negative for these observations. The adaptive-versus-matched-fixed interval includes zero, so a population-level inferiority claim at matched cost is not supported. Neither interval establishes how a larger or independently sampled query set would behave.

### Per-query controller choices and judged outcomes

| Query | ASR transcript | Confidence | Allocated budget | Used evidence tokens | MMR / adaptive / matched score |
| --- | --- | --- | --- | --- | --- |
| q_7e9a0f2716c017d2 | Understanding sales tax implications when purchasing items in a retail store. | 0.175 | 1780 | 1064 | 0.4167 / 0.4167 / 0.3333 |
| q_ea0be00eda6ea088 | Factors influencing the initial stock price at market opening. | 0.102 | 2042 | 1228 | 0.4167 / 0.3333 / 0.3333 |
| q_9cd791a970ab648f | Timeline for fund availability when depositing credit card payments. | 0.673 | 256 | 248 | 0.3333 / 0.0000 / 0.3333 |
| q_20e98c45013f6cc6 | Interpreting a negative price-to-earnings ratio for stocks. | 0.469 | 727 | 719 | 0.5833 / 0.4167 / 0.5833 |
| q_e3f1cef46917a2a5 | Process for adding an additional vehicle to an existing auto insurance policy with GICO. | 0.790 | 256 | 244 | 0.3333 / 0.0000 / 0.0000 |

Two queries receive budgets larger than the existing evidence and therefore have no evidence reduction. The controller aggressively reduces three others. The high-gap fund-availability query has no qrels-relevant document in the retrieved set and its adaptive answer scores zero, versus 0.3333 for unbudgeted and matched-fixed MMR. The example shows that a large gap does not establish that relevant evidence was retrieved; five observations cannot establish a general relationship between gap and answer quality. The observed score difference also includes independent generation variability.

Document-level relevant-document retention is **1.0 for four eligible queries in every arm**; the fifth has no relevant document before packing and is excluded from that retention mean. This does not establish sentence-level answer coverage: a relevant record can remain represented after its answer-bearing sentences are removed. It therefore does not contradict the lower answer scores.

### Rubric dimensions and per-query scores

| Condition | Correctness mean /4 | Grounding mean /4 | Completeness mean /4 | Mean score /1 |
| --- | --- | --- | --- | --- |
| simple_none | 1.20 | 1.00 | 0.80 | 0.2500 |
| tfidf_none | 1.20 | 1.20 | 0.80 | 0.2667 |
| mmr_none | 1.80 | 2.00 | 1.20 | 0.4167 |
| tfidf_fixed_512_skip | 1.20 | 1.20 | 0.60 | 0.2500 |
| mmr_fixed_512_skip | 1.60 | 1.20 | 1.20 | 0.3333 |
| mmr_adaptive_skip | 1.20 | 1.00 | 0.60 | 0.2333 |
| mmr_matched_skip | 1.40 | 1.40 | 1.00 | 0.3167 |
| mmr_fixed_256_skip | 1.20 | 1.00 | 1.00 | 0.2667 |
| mmr_fixed_1024_skip | 1.20 | 0.80 | 0.60 | 0.2167 |
| mmr_fixed_2048_skip | 2.40 | 2.40 | 1.80 | 0.5500 |

Query order Q1–Q5 is the order in the preceding transcript/controller table. These are the same five questions in every arm.

| Condition | Q1 | Q2 | Q3 | Q4 | Q5 |
| --- | --- | --- | --- | --- | --- |
| simple_none | 0.0000 | 0.3333 | 0.0000 | 0.5000 | 0.4167 |
| tfidf_none | 0.5000 | 0.3333 | 0.0000 | 0.5000 | 0.0000 |
| mmr_none | 0.4167 | 0.4167 | 0.3333 | 0.5833 | 0.3333 |
| tfidf_fixed_512_skip | 0.4167 | 0.0000 | 0.0000 | 0.4167 | 0.4167 |
| mmr_fixed_512_skip | 0.3333 | 0.3333 | 0.0000 | 1.0000 | 0.0000 |
| mmr_adaptive_skip | 0.4167 | 0.3333 | 0.0000 | 0.4167 | 0.0000 |
| mmr_matched_skip | 0.3333 | 0.3333 | 0.3333 | 0.5833 | 0.0000 |
| mmr_fixed_256_skip | 0.3333 | 0.5833 | 0.0000 | 0.4167 | 0.0000 |
| mmr_fixed_1024_skip | 0.3333 | 0.3333 | 0.0000 | 0.4167 | 0.0000 |
| mmr_fixed_2048_skip | 0.4167 | 1.0000 | 0.4167 | 0.5000 | 0.4167 |

Sources: [full arm statistics](elm-20261115-02/budget_pilot/summary.csv), [all 425 paired metric comparisons](elm-20261115-02/budget_pilot/paired_summary.csv), [paired query records](elm-20261115-02/budget_pilot/paired_per_query.csv), [confidence groups](elm-20261115-02/budget_pilot/query_groups.csv), [paired confidence groups](elm-20261115-02/budget_pilot/paired_groups.csv), [pre-budget context lengths](elm-20261115-02/inputs/budget_practice5.lengths.csv), and per-arm `metrics.jsonl`, `candidates.jsonl` and `judge/` records. There are **59 entries in the shared judge cache** supporting 70 condition/query score rows; repeated cache hits are not independent judgements. Retry HTTP attempts are not counted by those totals.

## 7. ASR validation and profiling

The supplied validation ran on **H200 MIG 3g.71gb, 60 SMs, sm_90**. Both model modes produced “Concord returned to its place amidst the tents.” The printed 100% measure is expected-word-set coverage on a single 3.5-second clip, not corpus WER. The archive records three timed transcriptions per mode after warmup, with model loading excluded.

| Task | Cases | Status | Maximum reported error |
| --- | --- | --- | --- |
| Task 1 | 4 | PASS | 1.91e-07 |
| Task 2 | 11 | PASS | 5.42e-07 |
| Task 3 | 10 | PASS | 3.53e-05 |
| Task 4 | 8 | PASS | 5.21e-07 |

| Mode | Status | Transcription ms, mean ± SD | ms/token | Peak allocated GiB | Single-clip word-set coverage |
| --- | --- | --- | --- | --- | --- |
| none | PASS | 419.0 ± 3.1 | 32.2 | 8.90 | 100.0% |
| all | PASS | 1638.2 ± 0.1 | 126.0 | 8.56 | 100.0% |

CUDA-all is **3.91× slower**, or **0.256× speedup**, and reduces peak allocated memory by **0.34 GiB** at the printed precision. The instrumented linear operations dominate the CUDA profile (**1,429.1 ms, 84.3%** of kernel time). RoPE improves in that profile, while attention and RMSNorm are slower. This motivates shape-aware GEMM optimisation, but does not by itself identify a single hardware cause.

The profile is a separate instrumented transcription; its totals should not be added to or equated with the mean wall-clock transcription above. Imported manual logs did not record historical full-source hashes, shell exit codes or a complete clean-build log for that validation. Current source hashes identify the continuation's code, not retrospectively every earlier execution. A later RAG preparation used a smaller 16-SM H200 MIG slice, so do not transfer the ASR latency numbers to that allocation.

| Operation | PyTorch kernel total ms | CUDA kernel total ms | CUDA kernel-time share |
| --- | --- | --- | --- |
| linear | 328.5 | 1429.1 | 84.3% |
| attention | 73.4 | 198.6 | 11.7% |
| rope | 28.8 | 2.4 | 0.1% |
| rmsnorm | 18.8 | 58.4 | 3.4% |
| silu_mul | 3.2 | 2.6 | 0.2% |
| layernorm | 1.1 | 1.0 | 0.1% |
| embedding | 0.3 | 0.5 | 0.0% |
| softmax | 0.3 | 1.7 | 0.1% |
| im2col | 0.1 | 0.1 | 0.0% |

### All 33 supplied validation cases

| Task | Case | Status | Maximum error | Kernel ms | Reported throughput | GEMM naive / cuBLAS GFLOP/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | tiny (2 x 3 tokens, Hq=3, Hk=1) | PASS | 1.03e-07 | 0.002 | 0.8 GB/s | N/A |
| 1 | odd sizes, partial rotation | PASS | 1.02e-07 | 0.002 | 8.3 GB/s | N/A |
| 1 | audio encoder (partial, 32/64) | PASS | 1.91e-07 | 0.015 | 1004.5 GB/s | N/A |
| 1 | text decoder (GQA 16/4, full) | PASS | 1.73e-07 | 0.003 | 409.8 GB/s | N/A |
| 2 | rmsnorm, one short row | PASS | 8.27e-08 | 0.002 | 0.2 GB/s | N/A |
| 2 | rmsnorm, odd cols | PASS | 3.68e-07 | 0.003 | 12.7 GB/s | N/A |
| 2 | rmsnorm, text decoder | PASS | 4.35e-07 | 0.005 | 230.4 GB/s | N/A |
| 2 | layernorm, odd cols (example) | PASS | 3.91e-07 | 0.004 | 11.2 GB/s | N/A |
| 2 | layernorm, audio encoder | PASS | 5.42e-07 | 0.013 | 1178.1 GB/s | N/A |
| 2 | softmax, one short row | PASS | 1.11e-08 | 0.002 | 0.0 GB/s | N/A |
| 2 | softmax, odd cols | PASS | 1.04e-09 | 0.003 | 7.6 GB/s | N/A |
| 2 | softmax, large inputs (+-100) | PASS | 2.67e-08 | 0.004 | 4.5 GB/s | N/A |
| 2 | softmax, all negative (-200..-150) | PASS | 6.11e-09 | 0.002 | 2.0 GB/s | N/A |
| 2 | softmax, vocabulary row (59264) | PASS | 8.13e-11 | 0.096 | 4.9 GB/s | N/A |
| 2 | softmax, 64 x 4096 | PASS | 7.95e-10 | 0.007 | 290.6 GB/s | N/A |
| 3 | tiny 3 x 5, K = 7 | PASS | 9.76e-09 | 0.002 | 0.0 GFLOP/s | 0.0 / 0.0 |
| 3 | odd sizes 37 x 100, K = 33, bias | PASS | 1.55e-07 | 0.003 | 94.0 GFLOP/s | 76.0 / 43.0 |
| 3 | odd sizes, bias + GELU | PASS | 7.70e-07 | 0.005 | 197.0 GFLOP/s | 133.0 / 153.0 |
| 3 | decoder q_proj (64x2048x2048) | PASS | 1.05e-05 | 0.173 | 3098.0 GFLOP/s | 396.0 / 14168.0 |
| 3 | decoder gate_proj (64x6144x2048) | PASS | 1.14e-05 | 0.501 | 3215.0 GFLOP/s | 408.0 / 15961.0 |
| 3 | decoder down_proj (64x2048x6144) | PASS | 3.53e-05 | 0.574 | 2806.0 GFLOP/s | 386.0 / 17731.0 |
| 3 | lm_head, last token (1x59264x2048) | PASS | 1.24e-05 | 1.110 | 219.0 GFLOP/s | 211.0 / 1132.0 |
| 3 | encoder q_proj (1500x1280x1280) | PASS | 9.66e-06 | 1.423 | 3455.0 GFLOP/s | 414.0 / 21014.0 |
| 3 | encoder fc1 + GELU (1500x5120x1280) | PASS | 6.92e-07 | 5.600 | 3511.0 GFLOP/s | 416.0 / 23081.0 |
| 3 | conv2 via im2col (1500x1280x3840) | PASS | 2.74e-05 | 4.282 | 3443.0 GFLOP/s | 416.0 / 22569.0 |
| 4 | tiny, one tile (4 queries, 5 keys) | PASS | 1.11e-07 | 0.005 | 1.0 GFLOP/s | N/A |
| 4 | several tiles, peaky scores | PASS | 4.97e-07 | 0.014 | 16.0 GFLOP/s | N/A |
| 4 | GQA 4 heads / 2 kv heads | PASS | 1.30e-07 | 0.007 | 86.0 GFLOP/s | N/A |
| 4 | causal, seq_q = seq_k | PASS | 4.86e-07 | 0.028 | 90.0 GFLOP/s | N/A |
| 4 | causal, 5 new queries, 37 keys | PASS | 1.78e-07 | 0.008 | 24.0 GFLOP/s | N/A |
| 4 | causal, one new query | PASS | 3.39e-07 | 0.028 | 21.0 GFLOP/s | N/A |
| 4 | text decoder (GQA 16/4, causal) | PASS | 5.21e-07 | 0.027 | 633.0 GFLOP/s | N/A |
| 4 | audio encoder (20 heads, 1500) | PASS | 2.68e-07 | 5.773 | 1995.0 GFLOP/s | N/A |

Sources: [kernel log](elm-20261115-02/asr/asr_tests.log), [PyTorch model log](elm-20261115-02/asr/asr_model_none.log), [CUDA model log](elm-20261115-02/asr/asr_model_all.log), [all case measurements](elm-20261115-02/excel/asr_kernel_cases.csv), [model timings](elm-20261115-02/excel/asr_models.csv), [profile export](elm-20261115-02/excel/asr_profiles.csv), and [original interpreted validation archive](../hw1-asr/experiments/asr_validation_20261115/README.md). GEMM tiled timings may include bias/GELU while naive/cuBLAS comparators time GEMM only; avoid treating every ratio as an identical-epilogue comparison.

## 8. Earlier CUDA experiment results, kept separate from the RAG run

These are existing cluster measurements under `hw1-asr/experiments/`, not new measurements and not additional ELM evidence. Hardware differs between experiment families. Their filenames are archive identifiers; do not infer an unrecorded execution timestamp from a filename alone.

### Task 1: RoPE block size, fast math, fusion and precomputed trigonometry

The block-size study and fusion follow-up report a **whole H200, 132 SMs**. All six normal block sizes pass the supplied four cases in three recorded trials. The table reports mean ± sample SD effective GB/s across those three trials; tiny timings are heavily rounded and bandwidth is an effective logical-byte metric.

| Threads | Tiny GB/s | Odd sizes GB/s | Audio encoder GB/s | Text decoder GB/s |
| --- | --- | --- | --- | --- |
| 32 | 0.8 ± 0.1 | 7.8 ± 0.0 | 750.7 ± 0.9 | 345.9 ± 0.7 |
| 64 | 0.8 ± 0.1 | 7.9 ± 0.1 | 1339.5 ± 1.8 | 428.9 ± 0.2 |
| 128 | 0.8 ± 0.0 | 8.0 ± 0.1 | 1872.6 ± 5.7 | 494.9 ± 1.0 |
| 256 | 0.8 ± 0.1 | 8.0 ± 0.1 | 1867.2 ± 4.9 | 514.3 ± 2.9 |
| 512 | 0.7 ± 0.1 | 7.4 ± 0.1 | 1807.3 ± 11.4 | 478.7 ± 2.3 |
| 1024 | 0.8 ± 0.1 | 6.3 ± 0.0 | 1715.0 ± 3.3 | 484.2 ± 2.7 |

Normal 128 threads is fastest among these measured audio-encoder means; 256 is fastest for the decoder. **Fast math at 256 is invalid as a whole-suite optimisation:** the audio case fails in all three raw logs with maximum error **1.94e-04**. The CSV contains only its nine passing small/decoder records and omits those three failed audio rows, so a CSV-only pass count would be misleading. Retain normal-math results for correctness-preserving comparisons.

The follow-up directly compares fused Q/K work versus two launches, with computed versus precomputed trig values. Each variant has seven trials, each timing 396 RoPE calls. All four variants pass with maximum error **1.9885e-07**. Lookup table construction costs **0.205408 ms**, excluded from the per-call times below.

| Variant | Launches/call | Trials | ms/call, mean ± SD | Mean effective GB/s |
| --- | --- | --- | --- | --- |
| fused_compute | 1 | 7 | 0.00584209 ± 0.00000163 | 2629.20 |
| split_compute | 2 | 7 | 0.00838859 ± 0.00000340 | 1831.06 |
| fused_table | 1 | 7 | 0.00470353 ± 0.00000348 | 3265.64 |
| split_table | 2 | 7 | 0.00710197 ± 0.00001630 | 2162.79 |

Fusion reduces the measured launch/work overhead; table lookup also improves steady-state timing, but its build cost must be amortised and its reuse conditions stated. These are per-kernel observations on a different GPU allocation from the full ASR validation, not measured full-model speedups. Sources: [block-size measurements](../hw1-asr/experiments/task1/results/20261101_143054/results.csv), [raw fast-math failure](../hw1-asr/experiments/task1/results/20261101_143054/fast_math_256_trial2.log), [fusion measurements](../hw1-asr/experiments/task1/results/followup/measurements.csv), [follow-up metadata](../hw1-asr/experiments/task1/results/followup/metadata.log).

### Task 2: reduction block size, atomic RMSNorm and barrier diagnostic

The block-size study uses **H200 MIG 1g.18gb, 16 SMs**. All six variants pass all 11 supplied cases. These are single recorded timing batches per case/configuration; no independent-repeat SD is available. The table includes all 66 case/configuration latency observations in milliseconds.

| Case | 32 threads | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| rmsnorm, one short row | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 |
| rmsnorm, odd cols | 0.012 | 0.007 | 0.005 | 0.003 | 0.003 | 0.002 |
| rmsnorm, text decoder | 0.023 | 0.013 | 0.008 | 0.005 | 0.004 | 0.005 |
| layernorm, odd cols (example) | 0.014 | 0.008 | 0.005 | 0.004 | 0.003 | 0.003 |
| layernorm, audio encoder | 0.093 | 0.057 | 0.057 | 0.059 | 0.075 | 0.114 |
| softmax, one short row | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 |
| softmax, odd cols | 0.011 | 0.007 | 0.004 | 0.003 | 0.003 | 0.003 |
| softmax, large inputs (+-100) | 0.012 | 0.007 | 0.005 | 0.004 | 0.003 | 0.003 |
| softmax, all negative (-200..-150) | 0.005 | 0.003 | 0.003 | 0.003 | 0.002 | 0.002 |
| softmax, vocabulary row (59264) | 0.774 | 0.381 | 0.192 | 0.097 | 0.050 | 0.029 |
| softmax, 64 x 4096 | 0.040 | 0.021 | 0.012 | 0.009 | 0.008 | 0.009 |

The best block size depends on the shape: the vocabulary softmax improves from **0.774 ms at 32 threads to 0.029 ms at 1024**, while the large audio LayerNorm is fastest at **0.057 ms** for 64/128 threads and slows to **0.114 ms** at 1024. This supports shape-specific investigation rather than one universally best block size.

The later 256-thread atomic-RMSNorm variant passes all 11 cases, but the three RMSNorm cases are slower than the normal-256 baseline from the earlier run on the same MIG class. Cross-run timing variation remains a limitation; no end-to-end model benefit was measured.

| RMSNorm case | Normal-256 ms | Atomic-256 ms | Atomic / normal latency | Atomic max error |
| --- | --- | --- | --- | --- |
| rmsnorm, one short row | 0.002 | 0.004 | 2.00× | 8.27e-08 |
| rmsnorm, odd cols | 0.003 | 0.014 | 4.67× | 3.11e-07 |
| rmsnorm, text decoder | 0.005 | 0.017 | 3.40× | 9.73e-07 |

The missing-barrier diagnostic also passes the 11 ordinary numerical cases, **but racecheck reports one shared-memory error hazard**. Passing outputs in these runs do not justify removing the barrier. Intentionally unsafe variants are correctness diagnostics, not accepted optimisations. The archives `20261102_104503/results.csv` and `20261102_104835/results.csv` contain headers but **zero measurement rows**; they are not additional successful experiments.

Sources: [block-size results](../hw1-asr/experiments/task2/results/20261102_104419/results.csv), [atomic/barrier results](../hw1-asr/experiments/task2/results/20261102_105649/results.csv), [racecheck error](../hw1-asr/experiments/task2/results/20261102_105649/logs/race_missing_barrier_256_racecheck.log).

### Task 3: GEMM tile size, padding, loading and synchronisation

The archived study reports **H200 MIG 3g.71gb, 60 SMs**, FP32/pedantic cuBLAS with TF32 disabled. Valid configurations pass all ten supplied cases; maximum reported error is **3.53e-05**. Each timing is one recorded batch (3 warmups / 20 launches for tiled and cuBLAS), rather than independent statistical repeats. The production setting remains TILE=16, padded/coalesced W loading and both barriers.

| Experiment | Configuration | Case | Time (ms) | Throughput | Correct? | What it shows |
|---|---|---|---|---|---|---|
| Baseline | TILE=16, padded, coalesced | Encoder fc1 + GELU | 5.608 | 3506 GFLOP/s | 10/10 PASS | Reference for padding/W-load ablations |
| A | TILE=8 | Encoder fc1 / lm_head | 8.648 / 0.801 | 2273 / 303 GFLOP/s | 10/10 PASS | Best tested tile for lm_head; weakest for large encoder GEMMs |
| A | TILE=16, separate rebuild | Encoder fc1 / lm_head | 5.600 / 1.110 | 3511 / 219 GFLOP/s | 10/10 PASS | Balanced shared setting |
| A | TILE=32 | Encoder fc1 / lm_head | 4.695 / 1.921 | 4188 / 126 GFLOP/s | 10/10 PASS | Best encoder throughput; worst tested tile for lm_head |
| B | TILE=16, no padding | Encoder fc1 | 13.318 | 1476 GFLOP/s | 10/10 PASS | 2.37x baseline latency |
| C | TILE=16, strided W | lm_head | 1.617 | 150 GFLOP/s | 10/10 PASS | 45.4% more latency than baseline |
| D | Second barrier removed | Seven large cases | Not used | Invalid results | 7/10 FAIL in each of 3 runs | Racecheck confirms shared-memory races |
| E | Baseline tiled / cuBLAS | lm_head | 1.112 / ~0.21406 | 436.6 / 2268 GB/s | PASS | Weight-byte effective bandwidth |
| Final | TILE=16, padded, coalesced | Encoder fc1 | 5.600 | 3511 GFLOP/s | 10/10 PASS | Production configuration verified |

#### All large-shape tile-size comparisons (archived values)

| Case (M,N,K) | TILE=8: ms / GFLOP/s | TILE=16: ms / GFLOP/s | TILE=32: ms / GFLOP/s | Fastest tested tile |
|---|---|---|---|---|
| decoder q_proj (64x2048x2048) | 0.291 / 1846 | 0.173 / 3098 | 0.185 / 2899 | 16 |
| decoder gate_proj (64x6144x2048) | 0.741 / 2172 | 0.501 / 3215 | 0.454 / 3548 | 32 |
| decoder down_proj (64x2048x6144) | 0.858 / 1878 | 0.570 / 2826 | 0.613 / 2627 | 16 |
| lm_head, last token (1x59264x2048) | 0.801 / 303 | 1.110 / 219 | 1.921 / 126 | 8 |
| encoder q_proj (1500x1280x1280) | 2.181 / 2254 | 1.423 / 3455 | 1.190 / 4130 | 32 |
| encoder fc1 + GELU (1500x5120x1280) | 8.648 / 2273 | 5.600 / 3511 | 4.695 / 4188 | 32 |
| conv2 via im2col (1500x1280x3840) | 6.545 / 2253 | 4.282 / 3444 | 3.592 / 4105 | 32 |

#### All large-shape padding comparisons (archived values)

| Case | Padded ms | Unpadded ms | Latency ratio | GFLOP/s: padded → unpadded | Throughput lost |
|---|---|---|---|---|---|
| Decoder q_proj | 0.174 | 0.394 | 2.26x | 3091 → 1363 | 55.9% |
| Decoder gate_proj | 0.502 | 1.138 | 2.27x | 3209 → 1415 | 55.9% |
| Decoder down_proj | 0.571 | 1.317 | 2.31x | 2822 → 1223 | 56.7% |
| lm_head | 1.112 | 2.643 | 2.38x | 218 → 92 | 57.8% |
| Encoder q_proj | 1.425 | 3.354 | 2.35x | 3450 → 1465 | 57.5% |
| Encoder fc1 + GELU | 5.608 | 13.318 | 2.37x | 3506 → 1476 | 57.9% |
| Conv2 via im2col | 4.282 | 10.059 | 2.35x | 3444 → 1466 | 57.4% |

#### All large-shape coalescing comparisons (archived values)

| Case | Coalesced ms | Strided ms | Extra latency | GFLOP/s: coalesced → strided | Throughput lost |
|---|---|---|---|---|---|
| Decoder q_proj | 0.174 | 0.247 | 42.0% | 3091 → 2177 | 29.6% |
| Decoder gate_proj | 0.502 | 0.713 | 42.0% | 3209 → 2258 | 29.6% |
| Decoder down_proj | 0.571 | 0.877 | 53.6% | 2822 → 1836 | 34.9% |
| lm_head | 1.112 | 1.617 | 45.4% | 218 → 150 | 31.2% |
| Encoder q_proj | 1.425 | 2.069 | 45.2% | 3450 → 2376 | 31.1% |
| Encoder fc1 + GELU | 5.608 | 8.196 | 46.1% | 3506 → 2399 | 31.6% |
| Conv2 via im2col | 4.282 | 6.204 | 44.9% | 3444 → 2377 | 31.0% |

TILE=32 improves the tested large encoder throughput by about **19%** relative to TILE=16, but makes the single-row lm_head slower; TILE=8 is fastest for that shape. Removing shared-memory padding loses **55.9–57.9%** throughput. Strided loading adds **42.0–53.6%** latency. These support keeping padding/coalesced loads, while tile selection remains shape-dependent.

Removing the second barrier causes **7/10 cases to fail in each of three runs**; racecheck reports four displayed hazards (two errors/two warnings). The final valid configuration restores both barriers. No clean production-kernel racecheck log is in this archive. Invalid-kernel timings are not speedup evidence.

For the initial baseline lm_head, weight-byte effective bandwidth is **436.6 GB/s tiled versus approximately 2,268 GB/s cuBLAS**. This is a logical weight-byte/time estimate, not measured DRAM traffic. Final encoder fc1 throughput is **3,511 GFLOP/s**, about **8.44×** the supplied naive kernel, while cuBLAS is **6.58×** faster than the tiled implementation for that case. No unfused GELU timing ablation was run, so a GELU-fusion runtime speedup is unavailable.

Source: [complete GEMM report, including every small-case record, barrier failures and all 70 passing case records](../hw1-asr/experiments/task3_gemm/results/RESULTS.md). Different baseline/rebuild/final runs are distinguished there; do not mix them when calculating an ablation ratio. These experiments do not establish a full-model optimal TILE.

### Task 4 coverage

The supplied Task 4 correctness/performance cases and CUDA-all integration are included in section 7. No separate Task 4 parameter-ablation measurement archive was found, so there is no additional Task 4 tuning result to consolidate.

## 9. Earlier synthetic RAG fixtures — separate from course/ELM results

The repository also contains two older fixture studies under `measurements/local_cpu_20261111/`. They ran on an Apple M4/macOS CPU with Python 3.11.11. They are included here for completeness, **not pooled with the university cluster results**. Their old findings documents predate the real ELM run; historical statements there that ELM quality was unavailable are superseded by sections 5–6 above.

### Synthetic dynamic-corpus study

The source is 24,000 normalised Gaussian **64-dimensional synthetic vectors**, not the **384-dimensional enc_a course vectors**. The same 21-case, three-strategy, three-repeat structure gives 189 repeat summaries. The principal fixture has N=2,000, 1,600 initially active records, 200 operations, batch=1 and one read/operation.

| Strategy | Mutation ms, mean ± SD | Query ms, mean ± SD | Workload ms, mean ± SD | Mutation speedup | Workload speedup |
| --- | --- | --- | --- | --- | --- |
| full_rebuild | 179.957 ± 2.483 | 0.0397 ± 0.0023 | 187.889 ± 2.774 | 1.00× | 1.00× |
| incremental | 14.936 ± 0.689 | 0.4162 ± 0.0210 | 98.171 ± 4.888 | 12.05× | 1.91× |
| incremental_compact | 14.742 ± 0.239 | 0.2383 ± 0.0028 | 62.399 ± 0.738 | 12.21× | 3.01× |

All synthetic case configurations are retained below. Times are means across three repeats; R = rebuild, I = incremental, C = incremental with compaction. These values describe this Mac fixture only.

| Synthetic case | Mutation ms R / I / C | Query ms R / I / C | Workload ms R / I / C | Workload speedup I / C |
| --- | --- | --- | --- | --- |
| principal: N=2000, ops=200, batch=1, mixed, reads/op=1, threshold=5% | 179.957 / 14.936 / 14.742 | 0.0397 / 0.4162 / 0.2383 | 187.889 / 98.171 / 62.399 | 1.91× / 3.01× |
| batch_size: N=2000, ops=500, batch=1, mixed, reads/op=1, threshold=5% | 457.425 / 42.061 / 38.640 | 0.0400 / 0.9412 / 0.2550 | 477.423 / 512.680 / 166.140 | 0.93× / 2.87× |
| batch_size: N=2000, ops=500, batch=10, mixed, reads/op=1, threshold=5% | 49.998 / 6.961 / 10.801 | 0.0339 / 0.2026 / 0.0726 | 66.964 / 108.282 / 47.093 | 0.62× / 1.42× |
| batch_size: N=2000, ops=500, batch=100, mixed, reads/op=1, threshold=5% | 8.348 / 3.596 / 5.420 | 0.0301 / 0.0534 / 0.0346 | 23.417 / 30.297 / 22.742 | 0.77× / 1.03× |
| batch_size: N=2000, ops=500, batch=500, mixed, reads/op=1, threshold=5% | 4.218 / 3.237 / 4.189 | 0.0297 / 0.0407 / 0.0299 | 19.045 / 23.570 / 19.116 | 0.81× / 1.00× |
| mutation_type: N=2000, ops=100, batch=1, insert, reads/op=1, threshold=5% | 95.319 / 7.571 / 8.402 | 0.0398 / 0.3237 / 0.2448 | 99.298 / 39.940 / 32.881 | 2.49× / 3.02× |
| mutation_type: N=2000, ops=100, batch=1, update, reads/op=1, threshold=5% | 91.728 / 7.497 / 8.096 | 0.0338 / 0.3280 / 0.2318 | 95.112 / 40.295 / 31.277 | 2.36× / 3.04× |
| mutation_type: N=2000, ops=100, batch=1, delete, reads/op=1, threshold=5% | 88.726 / 5.313 / 6.188 | 0.0389 / 0.0368 / 0.0361 | 92.619 / 8.992 / 9.795 | 10.30× / 9.46× |
| mutation_type: N=2000, ops=100, batch=1, mixed, reads/op=1, threshold=5% | 93.492 / 6.705 / 6.899 | 0.0393 / 0.2298 / 0.2287 | 97.424 / 29.687 / 29.770 | 3.28× / 3.27× |
| accumulation: N=2000, ops=300, batch=10, mixed, reads/op=1, threshold=5% | 31.033 / 4.369 / 6.034 | 0.0343 / 0.1420 / 0.0713 | 41.328 / 46.972 / 27.429 | 0.88× / 1.51× |
| threshold: N=2000, ops=300, batch=10, mixed, reads/op=1, threshold=1% | 31.365 / 4.640 / 13.276 | 0.0345 / 0.1438 / 0.0414 | 41.718 / 47.786 / 25.686 | 0.87× / 1.62× |
| threshold: N=2000, ops=300, batch=10, mixed, reads/op=1, threshold=5% | 31.154 / 4.252 / 5.970 | 0.0348 / 0.1426 / 0.0707 | 41.588 / 47.021 / 27.191 | 0.88× / 1.53× |
| threshold: N=2000, ops=300, batch=10, mixed, reads/op=1, threshold=10% | 31.621 / 4.200 / 5.508 | 0.0348 / 0.1420 / 0.1055 | 42.064 / 46.796 / 37.150 | 0.90× / 1.13× |
| threshold: N=2000, ops=300, batch=10, mixed, reads/op=1, threshold=20% | 31.520 / 4.160 / 4.239 | 0.0353 / 0.1406 / 0.1432 | 42.104 / 46.352 / 47.205 | 0.91× / 0.89× |
| query_ratio: N=2000, ops=10, batch=1, mixed, reads/op=1, threshold=5% | 9.478 / 0.685 / 0.671 | 0.0414 / 0.0632 / 0.0614 | 9.893 / 1.317 / 1.285 | 7.51× / 7.70× |
| query_ratio: N=2000, ops=10, batch=1, mixed, reads/op=10, threshold=5% | 9.565 / 1.496 / 0.649 | 0.0317 / 0.0552 / 0.0556 | 12.739 / 7.016 / 6.209 | 1.82× / 2.05× |
| query_ratio: N=2000, ops=10, batch=1, mixed, reads/op=100, threshold=5% | 9.817 / 0.708 / 0.738 | 0.0303 / 0.0560 / 0.0571 | 40.123 / 56.668 / 57.828 | 0.71× / 0.69× |
| corpus_size: N=2000, ops=100, batch=10, mixed, reads/op=1, threshold=5% | 10.532 / 1.372 / 1.342 | 0.0336 / 0.0727 / 0.0716 | 13.887 / 8.639 / 8.501 | 1.61× / 1.63× |
| corpus_size: N=5000, ops=100, batch=10, mixed, reads/op=1, threshold=5% | 24.230 / 2.185 / 2.171 | 0.0538 / 0.0933 / 0.0929 | 29.613 / 11.519 / 11.457 | 2.57× / 2.58× |
| corpus_size: N=10000, ops=100, batch=10, mixed, reads/op=1, threshold=5% | 51.059 / 3.550 / 3.721 | 0.0997 / 0.1366 / 0.1391 | 61.030 / 17.212 / 17.634 | 3.55× / 3.46× |
| corpus_size: N=24000, ops=100, batch=10, mixed, reads/op=1, threshold=5% | 123.809 / 7.582 / 7.498 | 0.2100 / 0.2485 / 0.2480 | 144.805 / 32.434 / 32.300 | 4.46× / 4.48× |

Source: [complete synthetic dynamic metrics](../experiments/rag_snapshots/rag-dynamic-corpus/rag/extensions/dynamic_corpus/measurements/local_cpu_20261111/summary.csv), [synthetic exactness checks](../experiments/rag_snapshots/rag-dynamic-corpus/rag/extensions/dynamic_corpus/measurements/local_cpu_20261111/correctness_summary.json), [historical fixture interpretation](../experiments/rag_snapshots/rag-dynamic-corpus/rag/extensions/dynamic_corpus/report_findings.md). Different dimensions, machines and input distributions prevent interpreting the difference between the synthetic and course timings as a machine speedup.

### Synthetic budget-controller overhead study

This study uses already-ranked synthetic evidence, not course answers. There are 24 configurations: four policy/packing choices at 1, 5, 20, 100, 500 and 1,000 units. Each has five warmups and thirty timed calls (720 recorded timing samples overall). Both fixed and adaptive configurations use a **1,024-token bound**; adaptive still computes the score statistic. This isolates controller/packing work rather than testing query-dependent adaptive budgets or ELM quality. Construction is excluded; traced Python allocation is measured separately and is not RSS.

| Configuration | Evidence units | Repetitions | Mean ± SD ms | P95 ms | Python peak KiB |
| --- | --- | --- | --- | --- | --- |
| fixed_skip | 1 | 30 | 0.0047 ± 0.0011 | 0.0075 | 2.5 |
| fixed_sentence_prefix | 1 | 30 | 0.0042 ± 0.0001 | 0.0043 | 2.5 |
| adaptive_skip | 1 | 30 | 0.0054 ± 0.0005 | 0.0064 | 2.5 |
| adaptive_sentence_prefix | 1 | 30 | 0.0052 ± 0.0001 | 0.0054 | 2.5 |
| fixed_skip | 5 | 30 | 0.0145 ± 0.0008 | 0.0162 | 6.7 |
| fixed_sentence_prefix | 5 | 30 | 0.0131 ± 0.0003 | 0.0137 | 6.7 |
| adaptive_skip | 5 | 30 | 0.0139 ± 0.0002 | 0.0143 | 6.8 |
| adaptive_sentence_prefix | 5 | 30 | 0.0140 ± 0.0006 | 0.0143 | 6.8 |
| fixed_skip | 20 | 30 | 0.0468 ± 0.0010 | 0.0484 | 26.2 |
| fixed_sentence_prefix | 20 | 30 | 0.0460 ± 0.0002 | 0.0463 | 26.2 |
| adaptive_skip | 20 | 30 | 0.0860 ± 0.0769 | 0.2731 | 26.2 |
| adaptive_sentence_prefix | 20 | 30 | 0.0774 ± 0.0359 | 0.1366 | 26.2 |
| fixed_skip | 100 | 30 | 0.1631 ± 0.0587 | 0.3190 | 51.9 |
| fixed_sentence_prefix | 100 | 30 | 0.4162 ± 0.0174 | 0.4644 | 62.8 |
| adaptive_skip | 100 | 30 | 0.1418 ± 0.0106 | 0.1690 | 51.9 |
| adaptive_sentence_prefix | 100 | 30 | 0.4195 ± 0.0175 | 0.4648 | 62.8 |
| fixed_skip | 500 | 30 | 0.5540 ± 0.0143 | 0.5887 | 235.8 |
| fixed_sentence_prefix | 500 | 30 | 2.2219 ± 0.0326 | 2.2590 | 260.8 |
| adaptive_skip | 500 | 30 | 0.5567 ± 0.0068 | 0.5685 | 235.9 |
| adaptive_sentence_prefix | 500 | 30 | 2.2435 ± 0.0115 | 2.2576 | 260.3 |
| fixed_skip | 1000 | 30 | 1.0737 ± 0.0174 | 1.1191 | 442.0 |
| fixed_sentence_prefix | 1000 | 30 | 4.7641 ± 0.2624 | 5.1624 | 472.1 |
| adaptive_skip | 1000 | 30 | 1.1211 ± 0.1950 | 1.1157 | 442.1 |
| adaptive_sentence_prefix | 1000 | 30 | 4.6666 ± 0.0603 | 4.7245 | 473.0 |

At 1,000 synthetic units, skip packing costs about **1.1 ms** and sentence-prefix packing about **4.7 ms** on this fixture. Extra prefix-boundary/count attempts are consistent with that difference. These are not cluster controller timings or proof that adaptive allocation improves quality. Source: [all overhead aggregates](../experiments/rag_snapshots/rag-adaptive-context-budget/rag/extensions/adaptive_context_budget/measurements/local_cpu_20261111/overhead.csv), [720 individual samples](../experiments/rag_snapshots/rag-adaptive-context-budget/rag/extensions/adaptive_context_budget/measurements/local_cpu_20261111/overhead_samples.csv), [method/machine manifest](../experiments/rag_snapshots/rag-adaptive-context-budget/rag/extensions/adaptive_context_budget/measurements/local_cpu_20261111/benchmark_manifest.json), [historical overhead findings](../experiments/rag_snapshots/rag-adaptive-context-budget/rag/extensions/adaptive_context_budget/report_findings.md).

## 10. Available figures, missing measurements and report limits

### Existing budget figures

- Quality versus configured budget: [PNG](elm-20261115-02/budget_pilot/figures/quality_vs_budget.png) · [PDF](elm-20261115-02/budget_pilot/figures/quality_vs_budget.pdf).

- Quality versus estimated evidence used: [PNG](elm-20261115-02/budget_pilot/figures/quality_vs_tokens.png) · [PDF](elm-20261115-02/budget_pilot/figures/quality_vs_tokens.pdf).

- Condition-level quality: [PNG](elm-20261115-02/budget_pilot/figures/fixed_vs_adaptive.png) · [PDF](elm-20261115-02/budget_pilot/figures/fixed_vs_adaptive.pdf).

- API input tokens versus generation latency: [PNG](elm-20261115-02/budget_pilot/figures/prompt_tokens_vs_generation_latency.png) · [PDF](elm-20261115-02/budget_pilot/figures/prompt_tokens_vs_generation_latency.pdf).

- Evidence reduction versus relative quality reduction: [PNG](elm-20261115-02/budget_pilot/figures/token_vs_quality_reduction.png) · [PDF](elm-20261115-02/budget_pilot/figures/token_vs_quality_reduction.pdf).

Caption the 2048-token point as a saturation control. Do not infer a causal quality improvement from its higher score. Prefer the paired same-selector and matched-cost tables for attribution.

### Outputs not available in this downloaded run

- `budget_offline/`, `budget_adaptive128_1024/`, `budget_adaptive512_2048/` and `budget_overhead/` are absent. Therefore this run has no separate `skip` versus `sentence_prefix` comparison, adaptive-range ablation or candidate-count overhead scaling result. Per-query controller time **is** measured in the ELM pilot (approximately 0.83–1.00 ms for enabled policies); that is different from the missing overhead sweep.
- The budget plot manifest explicitly skips `overhead_vs_candidates` because no overhead CSV was supplied.
- Dynamic raw data cover every case and repeat, but the downloaded `dynamic_course/figures/` has no PNG/PDF figures. The directory alone is not proof of generated plots.
- Earlier RAG extension fixtures under the repository's `measurements/local_cpu_20261111/` directories are **synthetic local CPU data**. They are not mixed into these course-vector/ELM tables. They remain separate historical fixtures rather than evidence of course answer quality or cluster performance.
- No full 350-query practice run, 125-query dev evaluation, repeated-generation experiment, repeated-judge uncertainty study, monetary-cost measurement or live audio-to-answer timing for these paired RAG arms is available.

### Limits that matter to the argument

The pilot uses the first five practice queries rather than a random sample, and contains one question with no relevant document retrieved at k=5. Generation uses temperature 0.1 and one answer per arm/query; judging uses one repeat with the same model family. Latency is network-dependent and output lengths vary. These conditions limit generalisation and causal attribution of small quality/timing differences.

Index quality is judged against identical active corpus states; corpus-subset/deletion effects must be reported as availability changes. The index benchmark uses in-memory precomputed compatible document pairs, deterministic refresh workloads and a simple full-rebuild control. It is not a comparison against the best database, an ANN engine or concurrent production traffic, and does not measure persistence/crash durability. Matrix MiB excludes RSS and retained snapshots.

## 11. Report-ready interpretation

**Dynamic corpus paragraph:** “We compared an exact base-plus-delta index, with and without automatic compaction, against full reconstruction after each atomic mutation batch. In the principal 200-operation workload, automatic compaction reduced maintenance time by 15.04× and combined update/query service time by 3.76×, while all 6,075 checkpoint/query checks across the suite matched a rebuilt reference. The improvement was workload-dependent: query latency increased in the principal case, batching reduced the rebuild disadvantage, and the most read-heavy configuration favoured rebuilding. Compaction therefore trades periodic maintenance for control of read overhead rather than providing a universal search speedup.”

**Adaptive context paragraph:** “We held retrieval, transcripts and MMR sentence selection fixed and compared unbudgeted, fixed-budget and retrieval-gap-adaptive evidence packing over five practice queries. Adaptive packing reduced mean API input tokens by 21.61%, but its mean normalised answer score decreased from 0.4167 to 0.2333. At matched mean evidence cost, fixed packing scored 0.3167; the paired adaptive-minus-fixed interval crossed zero. The pilot therefore demonstrates token reduction but does not support a quality-preserving advantage for the proposed adaptive rule. Identical prompts in the 2048-token saturation control produced different answers and scores, highlighting generation variability and the need for larger/repeated evaluation before policy selection.”

**ASR paragraph:** “All 33 supplied kernel cases and both model integration checks passed on the recorded H200 MIG allocation. CUDA-all required 1,638.2 ms per transcription versus 419.0 ms for PyTorch, despite reducing peak allocated memory from 8.90 to 8.56 GiB. Profiling attributed 84.3% of CUDA kernel time to linear operations. The result establishes numerical/integration correctness and identifies remaining optimisation work, rather than demonstrating model-level acceleration.”

A defensible report explains a measured tradeoff and a negative result. It should not claim that every extension improved both speed and quality. For a short report, prioritise the principal dynamic table plus batch/read-ratio results, the matched-cost budget comparison and saturation control, and the ASR correctness/model comparison. Keep exhaustive case tables and raw evidence in an appendix or this results bundle.

## 12. Source and Excel index

| Source | Use |
| --- | --- |
| [SUMMARY.md](elm-20261115-02/SUMMARY.md) | Original auto-exported run summary; preserved unchanged |
| [excel/asr_checks.csv](elm-20261115-02/excel/asr_checks.csv) | Task status, counts and numerical errors |
| [excel/asr_models.csv](elm-20261115-02/excel/asr_models.csv) | ASR model timing, memory and output |
| [excel/asr_kernel_cases.csv](elm-20261115-02/excel/asr_kernel_cases.csv) | All 33 kernel cases and GEMM comparators |
| [excel/asr_profiles.csv](elm-20261115-02/excel/asr_profiles.csv) | Operation profile totals |
| [excel/dynamic_systems.csv](elm-20261115-02/excel/dynamic_systems.csv) | 63 case/strategy systems summaries |
| [excel/dynamic_retrieval.csv](elm-20261115-02/excel/dynamic_retrieval.csv) | 63 final-state quality rows, availability and eligible counts |
| [excel/dynamic_answers.csv](elm-20261115-02/excel/dynamic_answers.csv) | Four real-ELM trial aggregate rows |
| [excel/budget_conditions.csv](elm-20261115-02/excel/budget_conditions.csv) | Ten budget condition summaries |
| [excel/budget_paired.csv](elm-20261115-02/excel/budget_paired.csv) | 425 paired metric comparisons |
| [dynamic_course/summary.csv](elm-20261115-02/dynamic_course/summary.csv) | Full systems metrics including SD and maintenance breakdown |
| [dynamic_course/all_summary_per_repeat.csv](elm-20261115-02/dynamic_course/all_summary_per_repeat.csv) | 189 repeat-level systems records |
| [dynamic_trial/rag_comparison.csv](elm-20261115-02/dynamic_trial/rag_comparison.csv) | All 20 dynamic condition/query records |
| [budget_pilot/summary.csv](elm-20261115-02/budget_pilot/summary.csv) | Full budget costs, rubric dimensions and sample counts |
| [budget_pilot/paired_per_query.csv](elm-20261115-02/budget_pilot/paired_per_query.csv) | Query-level paired differences |
| [inputs/practice5_transcripts.jsonl](elm-20261115-02/inputs/practice5_transcripts.jsonl) | Shared real ASR transcripts |
| [inputs/budget_practice5.meta.json](elm-20261115-02/inputs/budget_practice5.meta.json) | Frozen-input parameters and hashes |
| [source_sha256.json](elm-20261115-02/source_sha256.json) | Actual code-file fingerprints for this continuation |
| [commands.jsonl](elm-20261115-02/commands.jsonl) | Recorded subprocess arguments, exits and wall times |
| [stage_failures.jsonl](elm-20261115-02/stage_failures.jsonl) | Historical failures retained alongside later successful results |

Excel fraction columns are 0–1 and should use Percentage formatting. Columns ending in `_pct` are already 0–100. Blank/N/A cells mean undefined or unmeasured, not zero. Source measurements were rounded only for presentation here; compute further comparisons from the linked CSV/JSON rather than the printed tables.

## 13. Report emphasis after the coursework clarification

Planning update: **17 November 2026**, based on the Course Team email pasted by the user. The email's send date was not supplied. This is a recommended structure, not a claim about undisclosed marking weights. Experimental results above are unchanged.

### What the clarification changes

- Select and answer **one question from each of the four ASR tasks**. All four answers must appear somewhere in the report and be easy to locate.
- The ASR “Try This” sections are optional. Existing tuning experiments are useful supporting evidence; completing every suggested experiment is unnecessary.
- Relevant `make test` and model profiling results can be summarised in the main body, with remaining output in the appendix.
- Supplementary code can include the files necessary/useful to demonstrate the implemented system. Unchanged supplied framework files need not be resubmitted.
- The two video questions are to be specified in a separate announcement. Their contents are not established by this email.

The local [report guide](../report/README.md) still states a **six-page main-body limit**, with references and appendices excluded. The clarification does not change that limit. Keep the essential investigation, methods, comparisons and findings in the main body; use appendices for the explicitly permitted ASR answers and detailed supporting material.

### Recommended emphasis

**Make dynamic corpus maintenance the main systems investigation.** It has the clearest controlled comparison, complete course-vector systems measurements, three repetitions per strategy, an independent exact reference, and ablations that explain both improvements and regressions.

Use this focused main question:

> When does exact incremental corpus maintenance outperform full index rebuilding in an audio-query RAG system, and how does compaction change the update–query tradeoff?

The broader question in section 2 can introduce both investigations, but the dynamic study should carry the main experimental argument. Describe adaptive context as a smaller complementary study of the token/quality tradeoff. The two mechanisms were tested separately.

The dynamic evidence worth emphasising is:

1. **Correctness and fair baseline:** compare rebuilding and incremental strategies on the same active corpus, mutation order, compatible enc_a vectors and cached query embeddings. Explain snapshot-consistent text/vector retrieval, tombstones and atomic batches briefly. Report zero mismatches across 6,075 checkpoint/query checks.
2. **Principal tradeoff:** at 2,000 source records and 200 individual mutations, automatic compaction reduces mutation time by 15.04× and update-plus-query service time by 3.76×, but increases average search latency from 0.181 to 0.527 ms. This is the main table, with repeat SDs.
3. **When the improvement shrinks or reverses:** batching helps the rebuild control, and the 100-reads-per-mutation case favours rebuilding. These findings explain when the design is useful rather than presenting only its best case.
4. **Scaling:** the separate 24,000-source-record, 100-mutation/batch=10 experiment gives a 3.27× workload speedup with automatic compaction. Distinguish this case from the smaller principal experiment.
5. **ELM integration and limits:** after-state dynamic and rebuilt retrieval/evidence match for all five queries. Their score of 0.2833 uses shared generated answers for identical prompts. None of the 20 mutations changed the five queries' retrieved IDs/scores, so this pilot establishes integration for these inputs, not improved answer quality from updates. The 2,000-record principal subset also has zero relevant-document availability for these five queries; use the same-state oracle for its correctness claim rather than its absolute qrels recall.

### Suggested six-page main-body allocation

These are approximate space budgets, including figures/tables, not required section lengths. Fit the title/abstract and the supplied template into the overall limit.

| Main-body content | Approximate pages | Most important evidence |
| --- | --- | --- |
| Introduction, motivation and research questions | 0.50 | Changing corpora and limited evidence create two different cost tradeoffs |
| Design and experimental method | 0.75 | Baseline, base/delta/compaction design, frozen inputs, CPU/GPU separation, timing definitions and sample sizes |
| ASR correctness and profile | 0.75 | 33/33 cases, both transcription checks, 419.0 versus 1,638.2 ms, GEMM's 84.3% kernel-time share |
| Dynamic corpus results and explanation | 2.25 | Principal table, batch-size/read-ratio plots, exactness checks and 24k scaling point |
| Adaptive context pilot | 1.25 | Unbudgeted MMR versus adaptive versus matched fixed budget, actual input usage, scores and saturation control |
| Discussion, limitations and conclusion | 0.50 | Workload dependence, five-query coverage, cached generation, negative findings and limits of generalisation |
| **Total** | **6.00** | References and appendices remain separate |

For the adaptive section, use a compact table with `mmr_none`, `mmr_adaptive_skip` and `mmr_matched_skip`: API input means 1,250.2 / 980.0 / 980.6 and score means 0.4167 / 0.2333 / 0.3167. State the 0.143% evidence-cost matching error and the matched-quality interval crossing zero. Include one sentence explaining that fixed 2048 sent the same prompts as unbudgeted MMR; its higher score is not evidence of a beneficial budget change. The pilot supports token savings, but has not demonstrated quality preservation.

For ASR, a combined correctness/model table and a short profile explanation are sufficient in the main body. The custom kernels' slower full-model time is an interpretable result, not a reason to omit the comparison. Optional new kernel tuning has lower priority than completing the report's argument and reproducible evidence.

### Four selected ASR questions and clear appendix locations

The question numbers below refer to the current local guides. Reconcile the question text if using a newer supplied version.

| Appendix label | Selected report question | Suggested answer and existing evidence |
| --- | --- | --- |
| A.1 — Task 1, Q3 | Why rotate Q and K in one launch rather than two? | One launch avoids a second dispatch per call. The model calls RoPE 396 times. In the archived whole-H200 follow-up, fused-compute time is approximately 0.005842 ms/call versus 0.008389 for split-compute, across seven trials. Keep that microbenchmark separate from the 60-SM full-model timing. |
| A.2 — Task 2, Q4 | What can happen between two warps without the final block-reduction barrier in LayerNorm? | A fast warp can overwrite shared `partial[]` values for the variance reduction while a slower warp still reads values from the mean reduction. The final barrier prevents this reuse race. The missing-barrier diagnostic passes ordinary numerical tests but racecheck reports a hazard, illustrating why pass outputs alone are insufficient. |
| A.3 — Task 3, Q2 | Why can cuBLAS outperform the tiled kernel? | One supported difference is shape-dependent tiling: the current kernel always uses TILE=16, while cuBLAS selects among tiling strategies. Our own TILE=8/16/32 results demonstrate the importance of shape. Explain the reuse/parallelism tradeoff and cite a primary source such as the [NVIDIA GEMM guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#typical-tile-dimensions-in-cublas-and-performance). Do not attribute the plain-FP32 test gap to TF32: the supplied GEMM test disables it. The exact cuBLAS kernels were not profiled. |
| A.4 — Task 4, Q1 | How much memory does the full encoder score matrix use, versus shared memory per block? | The score matrix uses `20 × 1500 × 1500 × 4 = 180,000,000` bytes, about 171.66 MiB. With head_dim=64, KV_TILE=32, ATTN_WARPS=4 and the padded K pitch, one custom block uses `4 × (32×65 + 32×64 + 4×64) = 17,536` shared-memory bytes, or 17.125 KiB. A whole-tensor global allocation and per-block shared allocation have different scopes; the latter is not total attention/model memory. |

Add an explicit locator in the introduction, for example:

> Answers to the four required ASR questions are provided in Appendix A.1–A.4, with one selected question from each task. Detailed test and profiling outputs are provided in Appendix B.

### What belongs in the appendix and supplement

- Appendix A: the four required ASR question answers, clearly labelled with task, question number and question text.
- Appendix B: the remaining `make test` and profiling outputs, GPU details, original transcription and numerical tolerances. Keep a compact validation summary in the main body.
- Appendix C: exhaustive dynamic ablations, state-history/memory details and additional CUDA tuning results. The main-body batch/read-ratio results should still explain the headline tradeoff.
- Appendix D: per-query ELM evidence, answers, judge coverage/settings, paired intervals and additional budget curves. Keep the five-query limitation and key quality findings in the main body.
- Supplementary code: the five edited ASR task files; both RAG extensions; changed shared supporting files required by those extensions; useful run/analysis helpers and concise reproduction instructions. Dependencies on unchanged framework code should be stated. Preserve the run's source fingerprints. Exclude credentials and avoid bundling large model/data payloads unnecessarily.

### Highest-value work remaining for report preparation

Generate the dynamic batch-size and read-ratio figures from the **existing recorded CSVs**, since the downloaded dynamic figure directory contains no plot files. Add the principal table and same-state exactness evidence. Write the four appendix answers and make their locations explicit. Preserve the completed experimental record; additional experiments should address a specific remaining research question rather than the now-optional “Try This” checklist.
