# ASR validation results

Source: complete cluster terminal output supplied by the user; archived verbatim in `cluster_output.log`.
Bundle date: 2026-11-15. Execution timestamp, Git commit and shell exit codes were not recorded in this pasted log.
Execution directory: `/home/s2427231/edin-mls-26-autumn/hw1-asr`.
GPU: NVIDIA H200 MIG 3g.71gb, sm_90, 60 SMs (as reported by the supplied tests).

## Supplied kernel tests

| Task | Cases | Result | Largest reported error |
|---|---:|---|---:|
| 1: RoPE | 4 | ALL PASS | 1.91e-07 |
| 2: reductions | 11 | ALL PASS | 5.42e-07 |
| 3: tiled GEMM | 10 | ALL PASS | 3.53e-05 |
| 4: attention | 8 | ALL PASS | 5.21e-07 |

33 supplied kernel cases reported PASS. This is numerical agreement within the supplied tolerances, not bitwise equality.

## Full-model checks

| Mode | Result | Mean ms | SD ms | ms/token | Peak allocated GiB | Expected-word-set coverage |
|---|---|---:|---:|---:|---:|---:|
| PyTorch reference | PASS | 419.0 | 3.1 | 32.2 | 8.90 | 100.0% |
| CUDA all | PASS | 1638.2 | 0.1 | 126.0 | 8.56 | 100.0% |

Both modes produced: “Concord returned to its place amidst the tents.”
The supplied sample is 3.50 seconds; each timed run generated 13 tokens. Three timed runs per mode, with the benchmark default warmup.
The printed accuracy is expected-word-set coverage on this one sample. It is not corpus WER or a general ASR accuracy evaluation.

## Interpretation

CUDA-all mean transcription time was 3.91× the PyTorch reference (speedup = 0.256×). Correctness passed; an end-to-end performance improvement was not achieved.
The separately instrumented CUDA profile attributes 1429.1 ms / 84.3% of kernel time to linear operations, versus 328.5 ms / 72.3% in the PyTorch profile. Linear/GEMM is the dominant measured cost.
RoPE was faster in the instrumented CUDA profile (2.4 vs 28.8 ms), while attention and RMSNorm were slower. Per-operation totals come from an additional profiled transcription and should not be equated with the mean wall-clock timings above.
Peak allocated GPU memory was 8.56 vs 8.90 GiB, a reduction of 0.34 GiB at the precision printed by the benchmark. This is allocator-reported peak allocation, not total device/process memory.
Model-loading times are recorded separately in the CSV and excluded from the transcription latency comparison. No load-time speedup is claimed.

## Files for Excel

- `kernel_summary.csv`: task-level PASS/count/error evidence.
- `kernel_cases.csv`: all supplied cases with kernel latency/throughput and GEMM naive/cuBLAS comparisons.
- `model_comparison.csv`: measured model timing, memory and transcription.
- `operation_profiles.csv`: measured per-operation kernel cost.
- `derived_comparison.csv`: ratios computed from the printed mean timings.

All measurements above come from the supplied log. No tests or benchmarks were executed locally to produce this summary.
