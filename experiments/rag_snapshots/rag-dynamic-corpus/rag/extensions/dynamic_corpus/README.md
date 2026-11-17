# Dynamic corpus and incremental exact retrieval

## Research question and scope

How do incremental insert/update/delete operations compare with reconstructing an
exact dense index, including query cost, memory and compaction? This extension
adds corpus maintenance; it reuses the supplied retriever, RAG pipeline, sentence
selectors, reranker, query encoder, generator and answer evaluator.

Work is isolated on `feat/rag-dynamic-corpus`, based on
`origin/feature/rag_retrieval_context` (`e3ed9ea`). That base contains TF-IDF, MMR,
cross-encoder reranking and the supplied answer evaluator. The adaptive-budget
extension is present in a separate, uncommitted sibling worktree; it is not
silently copied or assumed merged. Its real builder passes an additional
compatibility test when on `PYTHONPATH`.

## Architecture and API

The supplied static retriever stores a normalised FP32 matrix and a JSON row-ID
map; `CourseData` separately maps IDs to text. `src/retriever.py` is the only shared
source file changed: in-memory construction and shared exact scoring/ranking
helpers were added. Ordinary file-based static retrieval keeps BLAS scoring and
the original ranking defaults. Controlled dynamic/rebuild comparisons opt into
the same FP32 row reduction and descending-score/ascending-ID exact-tie policy.
This avoids BLAS roundoff changing ties when the matrix is split into chunks.
There is no quantisation or approximate nearest-neighbour search. Tests also
compare ordinary static results with the original BLAS/partition calculation.

`index.py` publishes immutable states containing a compact base matrix, validity
masks (tombstones), append-only delta chunks, latest-row mapping and current text.
Search scores every active base/delta partition, takes local candidates and merges
the global exact top-k. Superseded rows remain allocated until compaction, but are
masked out. Each atomic batch appends at most one vector chunk.

```python
from extensions.dynamic_corpus import DynamicCorpus, Mutation, CompactionPolicy

corpus = DynamicCorpus(records, vectors, record_ids, embedding_space="enc_a",
                       compaction=CompactionPolicy(delta_ratio=.05, tombstone_ratio=.05))
corpus.insert_document("new", text, compatible_normalised_vector)
corpus.update_document("new", replacement_text, replacement_vector)
corpus.apply_batch([Mutation("delete", "new")])
result = corpus.search(query_vector, top_k=10, embedding_space="enc_a")
document = corpus.get_document(result["record_ids"][0])
stats = corpus.get_stats()
corpus.compact()                 # Explicit maintenance, separately timed.
corpus.reset(records, vectors, record_ids)
```

Duplicate insertion and absent update/delete raise `KeyError`. Bad dimensions,
nonfinite/zero/unnormalised vectors, invalid IDs and declared space mismatches are
rejected. Normalisation on mutation is opt-in. Every active ID has exactly one
current text/vector pair. Batch operations have sequential semantics and publish
all-or-nothing, including if automatic compaction fails.

`FullRebuildCorpus` uses the same mutation validation and exact search, but
reconstructs the complete active matrix **once per atomic batch**. Batch size one
therefore rebuilds after every mutation. This is an in-memory control, not an
optimised database or repeated disk-loading baseline.

Automatic compaction runs when physical delta rows / active records reaches the
configured ratio **or** the fraction of invalid base rows reaches its ratio.
Physical delta rows include stale updates, so repeated updates cannot accumulate
unbounded garbage without triggering the rule. Explicit compaction clears all
delta rows/tombstones and preserves rankings/texts.

## Embedding and consistency contracts

The local `enc_a` model is documented as a query encoder. There is no verified
compatible document-embedding interface here. Callers must supply matching
document text and precomputed compatible embeddings; an equal dimension cannot
prove model compatibility. Experimental mutations reference original source
text/vector pairs by ID, never encode newly authored documents with the query
encoder, and never write course assets or qrels. `refresh` updates retain original
content; `replace` copies another source pair under a logical target ID, logs
provenance, and disables original-qrels quality interpretation.

Writers serialise staging and publication under a lock. Readers acquire one
immutable version briefly under that lock, then search it without holding the
lock. A completed write is visible to later retrieval. An in-flight request may
finish with its older, explicitly versioned evidence after a later deletion;
it cannot mix that version's scores with newer text. Old snapshots retain memory.
No persistence or crash durability is claimed. ASR/generator thread safety remains
the responsibility of those existing components.

## RAG integration

```python
from extensions.dynamic_corpus.pipeline import make_pipeline

pipeline = make_pipeline(asr=asr, encoder=encoder, retriever=static_retriever,
                         data=data, context_builder=builder, generator=generator,
                         reranker=reranker, dynamic_corpus=corpus)
answer = pipeline.answer_audio(audio_path, k=10, final_k=5)
```

Omit `dynamic_corpus` to receive the unchanged supplied `RAGPipeline`. Dynamic mode
wraps that same pipeline with request-specific retriever/data/builder adapters.
Both reranking and context building read the retrieval snapshot. Existing simple,
TF-IDF and MMR builders work directly. A budget builder exposing
`build_with_retrieval` receives scores too; no budget algorithm is reimplemented.
Answers include `corpus_version`, retrieval, prompt, contexts, evidence and timing.
The supplied evaluator accepts the candidate schema.

## Measurements and reproducibility

Run commands from this worktree's repository root, with the RAG Python environment
active. Python 3.10+, NumPy and optional plotting dependencies are required:

```bash
python -m pip install -r rag/requirements.txt -r rag/extensions/dynamic_corpus/requirements.txt
python -m unittest discover -s rag/tests -v

# All factorised CPU ablations; explicitly synthetic, with no course-quality claim.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=rag python -m extensions.dynamic_corpus ablate \
  --dataset synthetic --synthetic-records 24000 --dimension 64 \
  --probes 4 --repeats 3 --output rag/extensions/dynamic_corpus/measurements/my_synthetic_run --plot

# A single case; replace is optional and deliberately invalidates original qrels.
PYTHONPATH=rag python -m extensions.dynamic_corpus benchmark --dataset synthetic \
  --corpus-size 2000 --operations 200 --batch-size 10 --update-mode replace \
  --output rag/extensions/dynamic_corpus/measurements/replacements --plot

# Regenerate summaries and PNG/PDF figures from saved raw CSVs, without timing again.
PYTHONPATH=rag python -m extensions.dynamic_corpus plot \
  --output rag/extensions/dynamic_corpus/measurements/local_cpu_20261111

# Exact replay checks source/catalog and mutation-log fingerprints.
PYTHONPATH=rag python -m extensions.dynamic_corpus replay --dataset synthetic \
  --synthetic-records 24000 --dimension 64 \
  --case-dir rag/extensions/dynamic_corpus/measurements/local_cpu_20261111/principal_6979282d2e
```

For replay, substitute the actual `principal_*` directory from your run if its
configuration differs. Existing output directories are rejected to protect logs.

The suite has 21 cases, all at fixed k=10: batch sizes 1/10/100/500 (500
mutations), four mutation types (100), accumulation (300 in batches of 10),
thresholds 1/5/10/20% (300), queries per mutation 1/10/100 (10 mutations), source
subsets 2k/5k/10k/24k (100), and the principal comparison (200 individual
mutations). Unless varied, source subset=2k, initially active=80%, mixed ratio=
40% inserts/30% updates/30% deletes, seed=2026. The remaining 20% is the initial
update pool. Each condition uses identical vectors, initial IDs, ordered log,
queries, k and scorer. Strategy order rotates across three repeats; each has the
same query warmup. Queries cycle through four probes.

Timing uses `perf_counter`. Raw per-operation staging, whole-batch mutation,
rebuild/compaction and search wall times are separate. Staging samples exclude
shared batch overhead; single-operation wall latency is reported only for batches
of one. Amortised batch cost includes maintenance. Workload time is the sum of
mutation wall time and scheduled query wall time; oracle checks, logging, initial
construction and final diagnostic manual compaction are excluded and labelled.
It is index workload service time, not live RAG end-to-end latency.

Each checkpoint independently replays the logical source map, builds a fresh
exact reference, and compares full logical state, ordered IDs, scores (2e-6
absolute/relative tolerance), active membership and current text. Any mismatch
aborts the experiment. Current matrix/mask/ID-array bytes are exact; text payload
and shallow metadata estimates are separate. First-repeat `tracemalloc` replay
measures additional traced peak allocations without contaminating latency samples;
it is not total RSS, initial input memory or reader-held snapshot memory.

See [report_findings.md](report_findings.md) for actual local results and limitations.
`measurements/local_cpu_20261111/` contains config/fingerprint files, mutation logs,
query vectors, raw timings/comparisons, summary CSV, correctness JSON, principal
table and nine PNG/PDF figures. Per-repeat rows and sample standard deviations are
retained. These are **synthetic CPU measurements, not enc_a corpus results**.

## Course and small real RAG reproduction

Install the supplied assets using `rag/README.md` first. These commands use real
ASR transcripts and the actual local query encoder; no reference questions replace
ASR. A missing cache requires explicit `--transcribe-missing` and a configured ASR.

```bash
PYTHONPATH=rag python -m extensions.dynamic_corpus prepare-queries \
  --transcript-cache rag/results/transcript_cache_practice.jsonl \
  --split practice --limit 5 --device cpu --transcribe-missing \
  --output rag/extensions/dynamic_corpus/measurements/practice_queries.npz

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=rag python -m extensions.dynamic_corpus ablate --dataset course \
  --queries rag/extensions/dynamic_corpus/measurements/practice_queries.npz \
  --repeats 3 --output rag/extensions/dynamic_corpus/measurements/course_run --plot

# Real ELM calls, opt-in: at most five matched queries by default.
PYTHONPATH=rag python -m extensions.dynamic_corpus rag-trial --generate \
  --queries rag/extensions/dynamic_corpus/measurements/practice_queries.npz \
  --output rag/extensions/dynamic_corpus/measurements/course_trial --limit 5
```

The trial compares the full original static corpus, the 80% initial dynamic
corpus, the mutated dynamic corpus and a freshly rebuilt identical mutated corpus.
It uses cached **real** ASR/query-encoder outputs, so reported wall time covers
downstream processing, not fresh audio transcription/embedding. Identical prompts
reuse a generated answer; cache hits are labelled and do not demonstrate model
determinism. At most 20 unique generations are needed for five queries/four
conditions, usually fewer. No large API sweep runs automatically. Selectors are
`--selector simple|tfidf|mmr`; optional reranking requires
`--rerank --reranker-model <existing-local-model-directory>`.

Evaluate with the existing judge, sharing its cache to avoid judging identical
answers twice; copy per-condition scores before the next invocation overwrites
the judge's summary files:

```bash
for condition in original_static dynamic_before dynamic_after freshly_rebuilt_after; do
  python rag/scripts/evaluate_answers.py --split practice --allow-subset \
    --candidates "rag/extensions/dynamic_corpus/measurements/course_trial/${condition}.jsonl" \
    --out-dir rag/extensions/dynamic_corpus/measurements/course_trial/shared_judge || break
  mkdir -p "rag/extensions/dynamic_corpus/measurements/course_trial/judge_${condition}"
  cp rag/extensions/dynamic_corpus/measurements/course_trial/shared_judge/answer_judge_*.json* \
    "rag/extensions/dynamic_corpus/measurements/course_trial/judge_${condition}/"
done
PYTHONPATH=rag python -m extensions.dynamic_corpus summarise-trial \
  --output rag/extensions/dynamic_corpus/measurements/course_trial
```

This produces `rag_comparison.csv/json` with answers, IDs/evidence, retrieval and
downstream latencies, correctness, grounding, completeness and normalised score.
Absent judgements remain empty. Run the evaluator with `--dry-run` first if desired.

For unchanged original text, checkpoint CSVs report original-qrels Recall@10,
separate active-qrels Recall/MRR/NDCG@10, and relevant-document availability.
Deleting relevant documents changes availability; it is not an indexing error.
Original gold also limits interpretation of answer scores after deletion. Semantic
replacements suppress original-qrels metrics entirely. Synthetic vector probes
have no quality judgements and never receive invented retrieval/answer scores.

## Limits

Metadata copies are O(active IDs) per batch; incremental does not mean O(1).
Small delta chunks add Python/ranking overhead per query. Compaction holds the
writer lock and can delay snapshot acquisition; concurrency correctness is tested,
but concurrent service latency has not been benchmarked. Memory estimates exclude
retained old snapshots and allocator overhead. Rebuild cost depends on this simple
control's copying/validation, not the best possible maintenance implementation.
Neither synthetic 64-dimensional vectors nor local CPU timings establish a
course/cluster performance claim. Course assets, real query caches and ELM
dependencies/credentials were unavailable for the recorded run.
