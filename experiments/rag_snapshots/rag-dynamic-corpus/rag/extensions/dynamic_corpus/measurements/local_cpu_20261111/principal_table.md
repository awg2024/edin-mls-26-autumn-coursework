# Measured principal comparison

| Metric | Full rebuild | Incremental | Incremental + compaction |
|---|---:|---:|---:|
| Mean insert wall latency (ms) | 0.8985 | 0.0806 | 0.0728 |
| Mean update wall latency (ms) | 0.9005 | 0.0786 | 0.0898 |
| Mean delete wall latency (ms) | 0.9007 | 0.0629 | 0.0588 |
| Mean scheduled query latency (ms) | 0.0397 | 0.4162 | 0.2383 |
| Mean per-repeat p95 query latency (ms) | 0.0439 | 0.7299 | 0.4282 |
| Embedding storage before manual compaction (MiB) | 0.3955 | 0.4248 | 0.4065 |
| Total mutation + automatic maintenance (ms) | 179.9571 | 14.9362 | 14.7425 |
| Measured index workload time (ms) | 187.8889 | 98.1706 | 62.3992 |
| Exact top-k agreement | 1.0000 | 1.0000 | 1.0000 |

Values are means across repeats. Embedding bytes exclude text, metadata, old snapshots and process RSS.
The workload sum excludes validation, logging, initial construction and the final diagnostic manual compaction.
