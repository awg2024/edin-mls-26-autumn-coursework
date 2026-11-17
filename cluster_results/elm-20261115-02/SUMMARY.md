# Cluster validation summary

Generated from recorded outputs only. Missing values mean not measured/not run.
RAG is a five-practice-query pilot. It is not the teammate's 350-query K sweep.

## Recorded stage failures

| stage | error |
| --- | --- |
| prepare | prepare_queries exited 2; see /home/s2427231/mls-results/elm-20261115-02/logs/prepare_queries.log |
| generate | Configure ELM_API_KEY in /home/s2427231/edin-mls-26-autumn/rag/.env |
| generate | Log already exists; use a new run directory or preserve/rename the previous log: /home/s2427231/mls-results/elm-20261115-02/logs/budget_generate.log |

## ASR validation

| check | status | cases | failed_cases | max_error |
| --- | --- | --- | --- | --- |
| Clean build | NOT RECORDED | Not measured | Not measured | Not measured |
| Task 1 | PASS | 4 | 0 | 0.0000 |
| Task 2 | PASS | 11 | 0 | 0.0000 |
| Task 3 | PASS | 10 | 0 | 0.0000 |
| Task 4 | PASS | 8 | 0 | 0.0000 |
| Model none | PASS | Not measured | Not measured | Not measured |
| Model all | PASS | Not measured | Not measured | Not measured |

| mode | status | transcription_mean_ms | peak_allocated_GiB | expected_word_set_coverage_pct |
| --- | --- | --- | --- | --- |
| none | PASS | 419.0000 | 8.9000 | 100.0000 |
| all | PASS | 1638.2000 | 8.5600 | 100.0000 |

Model 'accuracy' is the supplied benchmark's expected-word-set coverage on one audio clip; it is not WER.
Both kernel tests and the model run must pass. Profiling kernel-time totals are separate from wall-clock latency.

ASR results above were imported from the existing manual logs. Shell exit codes, clean-build output and historical source hashes were not recorded in those logs; no ASR validation was rerun.

## Dynamic corpus: principal systems comparison

| strategy | maintenance_ms | query_mean_ms | workload_ms | current_embedding_MiB | topk_agreement_fraction |
| --- | --- | --- | --- | --- | --- |
| full_rebuild | 479.4093 | 0.1811 | 515.6364 | 2.3730 | 1.0000 |
| incremental | 31.2799 | 0.8884 | 208.9588 | 2.5488 | 1.0000 |
| incremental_compact | 31.8831 | 0.5270 | 137.2785 | 2.4390 | 1.0000 |

Matrix storage excludes text, metadata, reader-held snapshots and RSS. Workload sums exclude validation/IO and diagnostic manual compaction.

Recorded exactness checks:

```json
{
  "query_checkpoint_checks": 6075,
  "topk_mismatches": 0,
  "score_mismatches": 0,
  "logical_state_mismatches": 0,
  "inactive_or_stale_text_results": 0,
  "max_score_error": 0.0,
  "qrels_evaluated_checks": 666,
  "agreement_fraction": 1.0
}
```

## Budget: real ELM pilot

| condition | generated_queries | judged_queries | evidence_tokens_after | API_input_tokens | generation_ms | answer_score_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| simple_none | 5 | 5.0000 | 1209.2000 | 1178.4000 | 4098.1865 | 0.2500 |
| tfidf_none | 5 | 5.0000 | 921.8000 | 1250.2000 | 3688.2000 | 0.2667 |
| mmr_none | 5 | 5.0000 | 921.8000 | 1250.2000 | 4529.0596 | 0.4167 |
| tfidf_fixed_512_skip | 5 | 5.0000 | 480.6000 | 711.0000 | 3010.1835 | 0.2500 |
| mmr_fixed_512_skip | 5 | 5.0000 | 482.2000 | 713.8000 | 4635.8907 | 0.3333 |
| mmr_adaptive_skip | 5 | 5.0000 | 700.6000 | 980.0000 | 3570.2742 | 0.2333 |
| mmr_matched_skip | 5 | 5.0000 | 699.6000 | 980.6000 | 4936.6491 | 0.3167 |
| mmr_fixed_256_skip | 5 | 5.0000 | 244.0000 | 395.6000 | 2531.6078 | 0.2667 |
| mmr_fixed_1024_skip | 5 | 5.0000 | 835.4000 | 1140.8000 | 5476.2081 | 0.2167 |
| mmr_fixed_2048_skip | 5 | 5.0000 | 921.8000 | 1250.2000 | 4614.4014 | 0.5500 |

## Dynamic corpus: real ELM trial

| condition | queries | generation_cache_hits | judged_queries | retrieval_ms_mean | normalised_answer_score_mean |
| --- | --- | --- | --- | --- | --- |
| original_static | 5 | 0 | 5 | 9.1796 | 0.2500 |
| dynamic_before | 5 | 1 | 5 | 17.7353 | 0.2833 |
| dynamic_after | 5 | 5 | 5 | 8.3500 | 0.2833 |
| freshly_rebuilt_after | 5 | 5 | 5 | 7.7378 | 0.2833 |

## Interpretation

Fraction columns range from 0 to 1; use Excel Percentage formatting.
Columns ending in _pct already range from 0 to 100; use Number formatting.
Budget limits use the configured counter; defaults estimate UTF-8 bytes/4. API token counts are separate measurements.
RAG experiments reuse ASR/encoder inputs; downstream pipeline timings are not fresh audio-to-answer wall times.
Cached generation is marked explicitly. Blank answer scores are not zeros. Original gold has limitations after removing relevant documents.
