# Measurement archive

`local_cpu_20261111/` is an actually executed **synthetic CPU** suite: 21 cases,
three strategies, three repetitions, fixed k=10. No course/ELM quality results
are included. Configuration and interpretation are in the extension README and
`../report_findings.md`.

| Artifact | Contents |
|---|---|
| `environment.json` | Hardware, versions, flags, command, final source hashes, resource availability |
| `unit_tests.txt` | Isolated branch: 41 discovered, 40 pass, one optional budget check skips |
| `unit_tests_with_budget.txt` | Sibling budget extension on PYTHONPATH: 41 pass, no skips |
| `local_cpu_20261111_run.txt` | Complete console output from the 189 workload runs and plots |
| `course_attempt.txt` | Missing-assets error; exit 2, no course timings produced |
| `replay_principal.txt` | Replayed logical digest, stats and independent final-query checks |
| `local_cpu_20261111/experiment_manifest.json` | Dataset/probes, all case definitions, repeats, controls and timing scope |
| `local_cpu_20261111/all_summary_per_repeat.csv` | One row per case/strategy/repeat |
| `local_cpu_20261111/summary.csv`, `summary.json` | Means, sample SDs, sample counts and speedups |
| `local_cpu_20261111/correctness_summary.json` | 4,860 query/checkpoint checks; no mismatches, no judged queries |
| `local_cpu_20261111/principal_table.md` | Compact measured comparison |
| `local_cpu_20261111/figures/` | Nine PNG/PDF pairs and plot manifest |

Each `<family>_<configuration-hash>/` contains:

- `config.json`: initial IDs, catalog fingerprint, workload and scorer/tie policy.
- `mutations.jsonl`: ordered operations with source text/vector IDs; original assets are untouched.
- `queries.npz`: the exact query IDs/vectors used; source metadata is in `config.json`.
- `batch_timings.csv`: whole mutation wall time, staging, rebuild/compaction and amortised cost.
- `operation_timings.csv`: operation identity, raw staging time and single-op wall time where applicable.
- `query_timings.csv`: scheduled/diagnostic phase, version, query wall time, returned IDs and scores.
- `retrieval_checks.csv`: rebuilt-reference IDs/scores, active/current-text checks and quality scope.
- `state_history.csv`: checkpoints before/after updates/manual compaction, counts, ratios and memory estimates.
- `summary_per_repeat.csv`: separate mutation, query, workload and traced-memory summaries.

Repeated probe reads and oracle validation intentionally warm caches. Workload
sums exclude that diagnostic work, serialization, initial build and the final
manual compaction. Memory figures describe current index buffers, not whole
process RSS or the shared input pool. Missing relevance/answer judgements are
empty, not zero. Regeneration reads raw files; it does not invent or rerun timings.
