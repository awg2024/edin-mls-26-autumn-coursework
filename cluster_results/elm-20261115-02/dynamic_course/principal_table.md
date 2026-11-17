# Measured principal comparison

| Metric | Full rebuild | Incremental | Incremental + compaction |
|---|---:|---:|---:|
| Mean insert wall latency (ms) | 2.4607 | 0.1705 | 0.1576 |
| Mean update wall latency (ms) | 2.4099 | 0.1691 | 0.2083 |
| Mean delete wall latency (ms) | 2.2993 | 0.1249 | 0.1130 |
| Mean scheduled query latency (ms) | 0.1811 | 0.8884 | 0.5270 |
| Mean per-repeat p95 query latency (ms) | 0.2203 | 1.4943 | 0.8834 |
| Embedding storage before manual compaction (MiB) | 2.3730 | 2.5488 | 2.4390 |
| Total mutation + automatic maintenance (ms) | 479.4093 | 31.2799 | 31.8831 |
| Measured index workload time (ms) | 515.6364 | 208.9588 | 137.2785 |
| Exact top-k agreement | 1.0000 | 1.0000 | 1.0000 |

Values are means across repeats. Embedding bytes exclude text, metadata, old snapshots and process RSS.
The workload sum excludes validation, logging, initial construction and the final diagnostic manual compaction.
