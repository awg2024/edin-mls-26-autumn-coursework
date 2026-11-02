# Task 3 — Tiled GEMM results and interpretation

The implemented shared-memory CUDA linear layer passes all ten supplied Task 3 tests in its final production configuration. All seven valid configuration runs pass every case, including odd dimensions, optional bias and GELU. The maximum reported absolute error in each valid run is **3.53e-05**.

On this **NVIDIA H200 MIG 3g.71gb allocation**, TILE=32 gives the highest throughput for the tested large encoder GEMMs, approximately **19% above TILE=16**. TILE=8 is fastest for `lm_head`. Removing padding reduces large-case throughput by **55.9–57.9%**; strided W loading increases large-case latency by **42.0–53.6%**. Removing the second barrier produces incorrect results and shared-memory races. The final source retains TILE=16, padding, coalesced W loads and both barriers.

## Compact table for the PR/report

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

The first baseline, separate TILE=16 rebuild and final production run are distinct measurements of the same configuration. Padding and W-load comparisons below use the **initial baseline**; TILE comparisons use the **separate tile8/tile16/tile32 runs**. This avoids mixing reference runs when calculating differences.

## Hardware, software and methodology

| Item | Recorded value |
|---|---|
| Measurement date | 27 October 2026; environment captured at 15:01:52 UTC |
| Host platform | Linux x86_64, university teaching cluster |
| GPU allocation | NVIDIA H200 MIG 3g.71gb, 60 SMs, compute capability 9.0 (`sm_90`) |
| Approximate peak bandwidth printed by test | 2407 GB/s |
| CUDA compiler | CUDA 12.8, nvcc V12.8.93 |
| NVIDIA driver | 595.91.07; `nvidia-smi` displays CUDA 13.2 separately from the installed compiler version |
| Compiler flags | `-O3 -std=c++17 -lineinfo -diag-suppress 177 -arch=sm_90 -Xcompiler -fPIC` |
| Captured Git HEAD | `d0b99ee93fd3c246b621e122bbdc0d14d7cc4efa` |
| Data type / cuBLAS math mode | FP32 / `CUBLAS_PEDANTIC_MATH`, with TF32 disabled |
| Correctness references | Three small cases: CPU double-precision loop; seven large cases: cuBLAS; reference bias/GELU applied afterwards |
| Acceptance tolerance | Per element: `abs(error) <= 1e-3 + 1e-3 * abs(reference)`; nonfinite results fail |
| Tiled/cuBLAS timing | CUDA events; 3 warmups, 20 timed launches, mean time per launch |
| Naive timing | CUDA events; 1 warmup, 5 timed launches |
| Reported GFLOP/s | `2*M*N*K / (time_seconds * 1e9)` |

Sources: [environment capture](../raw_results/environment_gpu.txt), [baseline build](../raw_results/baseline_build.log), [test implementation](../../../tests/test_task3_gemm.cu), [timing helper](../../../tests/test_util.h) and [experiment commands](../README.md#commands-for-the-teaching-cluster).

Each valid variant was rebuilt in its own temporary source/build directory using the Makefile's `SRC` and `BUILD` overrides. Only TILE, W-tile padding, W-loading order or the second barrier changed in the corresponding experiment. The baseline is TILE=16 with `ws[16][17]`, coalesced W reads and both barriers.

These are per-launch microbenchmarks, with one recorded timing batch per ablation, rather than repeated independent statistical trials. Timings are printed to 0.001 ms and throughput to whole GFLOP/s, so derived ratios are approximate. No standard deviation or confidence interval is available. Tiny-case throughput printed as zero is a rounding effect, not evidence that no work occurred.

The tiled timing includes bias/GELU when requested; naive and cuBLAS comparison timings perform GEMM only. FLOP counts exclude epilogue arithmetic. The results do not measure model-level latency or prove a workload-wide optimal TILE.

## A — TILE size

All three configurations pass 10/10 cases.

| Case (M,N,K) | TILE=8: ms / GFLOP/s | TILE=16: ms / GFLOP/s | TILE=32: ms / GFLOP/s | Fastest tested tile |
|---|---|---|---|---|
| decoder q_proj (64x2048x2048) | 0.291 / 1846 | 0.173 / 3098 | 0.185 / 2899 | 16 |
| decoder gate_proj (64x6144x2048) | 0.741 / 2172 | 0.501 / 3215 | 0.454 / 3548 | 32 |
| decoder down_proj (64x2048x6144) | 0.858 / 1878 | 0.570 / 2826 | 0.613 / 2627 | 16 |
| lm_head, last token (1x59264x2048) | 0.801 / 303 | 1.110 / 219 | 1.921 / 126 | 8 |
| encoder q_proj (1500x1280x1280) | 2.181 / 2254 | 1.423 / 3455 | 1.190 / 4130 | 32 |
| encoder fc1 + GELU (1500x5120x1280) | 8.648 / 2273 | 5.600 / 3511 | 4.695 / 4188 | 32 |
| conv2 via im2col (1500x1280x3840) | 6.545 / 2253 | 4.282 / 3444 | 3.592 / 4105 | 32 |

**Large encoder GEMMs:** TILE=32 wins on encoder q_proj (4130 GFLOP/s), encoder fc1 + GELU (4188 GFLOP/s) and the conv2 GEMM (4105 GFLOP/s). Relative to the TILE=16 rebuild, throughput increases are 19.5%, 19.3% and 19.2% respectively. Larger tiles increase operand reuse and reduce K-loop barrier iterations. The measurements establish the net benefit here; no profiler counters were collected to separate reuse, occupancy or scheduling effects.

**Decoder shapes:** the winner depends on the shape. TILE=16 wins q_proj and down_proj, while TILE=32 wins gate_proj. A universal claim that larger tiles always help is not supported.

**lm_head:** TILE=8 takes 0.801 ms, TILE=16 takes 1.110 ms, and TILE=32 takes 1.921 ms. TILE=8 reduces latency by 27.8% relative to TILE=16, whereas TILE=32 takes 1.73x as long.

For M=1 and TILE=32, **only one of 32 thread rows** corresponds to a real output row: 32 of 1024 threads, or **3.125%**, produce valid outputs. The other 31 rows still execute loads, multiply-accumulates and barriers. They are not inactive CUDA lanes; their computed outputs are unused. Useful-output fractions are 12.5%, 6.25% and 3.125% for TILE=8, 16 and 32. This utilisation loss explains why the encoder's preferred tile is a poor choice for this single-row shape, although these measurements alone do not isolate every source of latency.

Sources: [TILE=8](../raw_results/tile8.log), [TILE=16](../raw_results/tile16.log), [TILE=32](../raw_results/tile32.log).

## B — Shared-memory padding

Only `ws[TILE][TILE + 1]` changes to `ws[TILE][TILE]`; TILE remains 16. Both versions pass 10/10 tests.

| Case | Padded ms | Unpadded ms | Latency ratio | GFLOP/s: padded → unpadded | Throughput lost |
|---|---|---|---|---|---|
| Decoder q_proj | 0.174 | 0.394 | 2.26x | 3091 → 1363 | 55.9% |
| Decoder gate_proj | 0.502 | 1.138 | 2.27x | 3209 → 1415 | 55.9% |
| Decoder down_proj | 0.571 | 1.317 | 2.31x | 2822 → 1223 | 56.7% |
| lm_head | 1.112 | 2.643 | 2.38x | 218 → 92 | 57.8% |
| Encoder q_proj | 1.425 | 3.354 | 2.35x | 3450 → 1465 | 57.5% |
| Encoder fc1 + GELU | 5.608 | 13.318 | 2.37x | 3506 → 1476 | 57.9% |
| Conv2 via im2col | 4.282 | 10.059 | 2.35x | 3444 → 1466 | 57.4% |

Latency ratio is `time_unpadded / time_baseline`. Throughput loss is `100 * (1 - GF_unpadded/GF_baseline)`.

Across the seven large cases, removing padding increases latency by **2.26–2.38x** and loses **55.9–57.9%** of throughput. Encoder fc1 rises from 5.608 to 13.318 ms and falls from 3506 to 1476 GFLOP/s. **Retaining the extra shared-memory column is strongly supported on this GPU.**

The inner loop reads `ws[tx][q]`: different output-column threads read a fixed K column across shared-memory rows. With stride 16 floats, the 16 distinct row addresses map to only two of 32 banks. Padding changes the row stride to 17 floats, distributing those distinct addresses across banks. The timing change is consistent with this bank-conflict mechanism; actual conflict counts were not profiled.

Sources: [padded baseline](../raw_results/baseline_tile16.log), [unpadded run](../raw_results/no_padding.log).

## C — Coalesced versus strided W loading

Both configurations use TILE=16, padding and both barriers, and pass 10/10 tests. The strided version swaps loader coordinates and shared-memory store coordinates together, preserving `ws[n][k] = W[blockIdx.x*TILE+n][k0+k]` and therefore the mathematical result.

| Case | Coalesced ms | Strided ms | Extra latency | GFLOP/s: coalesced → strided | Throughput lost |
|---|---|---|---|---|---|
| Decoder q_proj | 0.174 | 0.247 | 42.0% | 3091 → 2177 | 29.6% |
| Decoder gate_proj | 0.502 | 0.713 | 42.0% | 3209 → 2258 | 29.6% |
| Decoder down_proj | 0.571 | 0.877 | 53.6% | 2822 → 1836 | 34.9% |
| lm_head | 1.112 | 1.617 | 45.4% | 218 → 150 | 31.2% |
| Encoder q_proj | 1.425 | 2.069 | 45.2% | 3450 → 2376 | 31.1% |
| Encoder fc1 + GELU | 5.608 | 8.196 | 46.1% | 3506 → 2399 | 31.6% |
| Conv2 via im2col | 4.282 | 6.204 | 44.9% | 3444 → 2377 | 31.0% |

Extra latency is `100 * (time_strided/time_baseline - 1)`. Throughput loss is `100 * (1 - GF_strided/GF_baseline)`.

Strided loading adds **42.0–53.6% latency** across the large cases and loses **29.6–34.9% throughput**. For encoder fc1, time rises from 5.608 to 8.196 ms; for lm_head, from 1.112 to 1.617 ms. **Coalesced loading should remain in the final kernel.**

In the correct loader, adjacent tx threads read adjacent K elements in a W row. In the experimental loader, adjacent tx threads read different rows separated by K floats. The result is consistent with less efficient global-memory transactions. The loader change also changes shared-memory write ordering, and no transaction counters were collected, so the complete timing difference cannot be quantitatively assigned to global-memory coalescing alone.

Sources: [coalesced baseline](../raw_results/baseline_tile16.log), [strided run](../raw_results/strided_w.log).

## D — Synchronisation correctness diagnostic

The second `__syncthreads()` was removed only from the temporary diagnostic copy. In **all three regular runs**, the three small cases pass and **all seven large cases fail**.

The following cells give **incorrect output count / maximum absolute error**:

| Failing case | Run 1: wrong count / max error | Run 2: wrong count / max error | Run 3: wrong count / max error |
|---|---|---|---|
| Decoder q_proj | 22,789 / 6.25e-01 | 23,484 / 6.89e-01 | 24,177 / 7.99e-01 |
| Decoder gate_proj | 55,777 / 7.02e-01 | 55,356 / 6.51e-01 | 54,504 / 8.80e-01 |
| Decoder down_proj | 29,999 / 8.48e-01 | 29,397 / 9.99e-01 | 30,333 / 9.44e-01 |
| lm_head | 14,287 / 5.60e-01 | 14,135 / 6.22e-01 | 14,302 / 6.94e-01 |
| Encoder q_proj | 372,837 / 7.57e-01 | 371,010 / 6.97e-01 | 366,871 / 7.66e-01 |
| Encoder fc1 + GELU | 1,385,594 / 8.71e-01 | 1,381,380 / 1.04e+00 | 1,384,096 / 8.10e-01 |
| Conv2 via im2col | 680,380 / 1.08e+00 | 682,133 / 9.90e-01 | 687,419 / 9.21e-01 |

The failing-case set is repeatable in the three recorded runs, but error magnitudes, incorrect counts and first failing indices vary despite fixed input seeds. Thus the **numerical corruption varies between runs**. Three runs establish this observed variation, not a general probability of failure.

The make-wrapper exit status is **2** in each regular diagnostic run. Under:

```bash
compute-sanitizer --tool racecheck --error-exitcode=1 \
  /tmp/tmp.Fj9cV2olae/no_second_barrier/build/sm_90/test_task3_gemm small
```

racecheck reports **4 hazards displayed: 2 errors and 2 warnings**, with many individual read/write hazard instances. The recorded sanitizer exit status is **1**. It identifies shared-memory tile writes at source lines 36/46 racing with reads in the inner multiply loop at line 55. Under instrumentation, the small K=129 bias+GELU case also fails: **36 of 3465 values wrong**, maximum error **1.15e-01**. The other two small cases pass.

This demonstrates why the second barrier is needed: a faster warp must not overwrite the current tiles for K iteration i+1 while another warp still consumes iteration i. The first barrier ensures tile writes finish before consumption; the second finishes consumption before reuse. Threads outside Y still participate in loading tiles and must reach both barriers. Guard only the final epilogue/store.

The intentionally incorrect kernel's timing is not treated as a valid optimisation result. The final production run passes all cases with both barriers restored. **There is no production-kernel racecheck log in this dataset**, so no claim of a clean production sanitizer run is made.

Sources: [regular run 1](../raw_results/no_second_barrier.log), [run 2](../raw_results/no_second_barrier_repeat2.log), [run 3](../raw_results/no_second_barrier_repeat3.log), [racecheck](../raw_results/no_second_barrier_racecheck.log), and their adjacent `.exit_status` files.

## E — lm_head effective memory bandwidth

For M=1, N=59264, K=2048 and FP32:

```text
Weight bytes = N*K*4 = 485,490,688 bytes
             = 485.490688 MB = 0.485490688 GB (decimal) = 463 MiB
Effective bandwidth [GB/s] = 485.490688 / time_ms
```

This assumes one full weight read per invocation. The X array occupies 8192 bytes and the output 237056 bytes. This weight-only estimate excludes the output store and X reads, including the tiled kernel's repeated X loads across output-column blocks. It is **effective bandwidth**, not measured DRAM bandwidth: cache reuse, redundant traffic and transaction overhead can change physical memory traffic.

| Implementation/run | Time (ms) | Effective GB/s | Timing basis |
|---|---|---|---|
| Baseline tiled | 1.112 | 436.6 | Direct printed time |
| TILE=8 tiled | 0.801 | 606.1 | Direct printed time |
| TILE=16 repeat | 1.110 | 437.4 | Direct printed time |
| TILE=32 tiled | 1.921 | 252.7 | Direct printed time |
| No padding | 2.643 | 183.7 | Direct printed time |
| Strided W | 1.617 | 300.2 | Direct printed time |
| Final production tiled | 1.110 | 437.4 | Direct printed time |
| Baseline cuBLAS | ~0.21406 | 2268 | Inferred from 1134 GFLOP/s |
| Final cuBLAS | ~0.21425 | 2266 | Inferred from 1133 GFLOP/s |

The supplied tests print cuBLAS throughput but not cuBLAS latency. With `FLOPs = 2*M*N*K = 242,745,344` and printed cuBLAS throughput G:

```text
cuBLAS time_ms ~= 242.745344 / G
cuBLAS effective GB/s ~= 2*G
```

The required **initial baseline** result is approximately **436.6 GB/s tiled versus 2268 GB/s cuBLAS**. The final production run gives approximately **437.4 versus 2266 GB/s**. The small difference between these runs should not be interpreted as a measured optimisation.

GFLOP/s alone hides the poor M-dimension utilisation here. The tiled kernel's low useful-output fraction and wasted arithmetic reduce delivered bandwidth; the cuBLAS comparison shows this educational implementation leaves substantial performance available. TILE=8 is the best of the tested educational configurations for lm_head, at approximately 606.1 effective GB/s.

## Final production verification and configuration decision

Final configuration: **TILE=16; padded W tile; coalesced W reads; both barriers; zero-filled edge tiles; optional bias and GELU fused into a guarded epilogue.** The public launcher and supplied tests are unchanged. No intentionally degraded experimental variant remains in the source.

The recorded sequence includes `make clean`, followed by a successful `make`/Task 3 run. The final test ends with `Task 3 (tiled GEMM): ALL PASS`; all ten cases pass and maximum error is 3.53e-05.

| Representative case | Time (ms) | Tiled GFLOP/s | Naive GFLOP/s | cuBLAS GFLOP/s | Tiled / naive | cuBLAS / tiled |
|---|---|---|---|---|---|---|
| Decoder q_proj | 0.174 | 3093 | 396 | 14126 | 7.81x | 4.57x |
| Decoder gate_proj | 0.502 | 3207 | 406 | 15828 | 7.90x | 4.94x |
| Decoder down_proj | 0.572 | 2815 | 386 | 17803 | 7.29x | 6.32x |
| lm_head | 1.110 | 219 | 211 | 1133 | 1.04x | 5.17x |
| Encoder q_proj | 1.423 | 3455 | 415 | 21019 | 8.33x | 6.08x |
| Encoder fc1 + GELU | 5.600 | 3511 | 416 | 23087 | 8.44x | 6.58x |
| Conv2 via im2col | 4.282 | 3444 | 416 | 22574 | 8.28x | 6.55x |

TILE=16 remains the current production setting because it balances the tested shapes: TILE=32 improves large encoder throughput but makes lm_head approximately 73% slower than TILE=16, while TILE=8 improves lm_head but substantially slows the encoder GEMMs. **TILE=16 is not proven to be the fastest setting for the full model.** An encoder-heavy workload could benefit from TILE=32; a shape-dependent implementation could consider TILE=32 for large encoders and TILE=8 for M=1. Such changes require their own implementation and workload validation and are not part of this results-only update.

The final encoder fc1 achieves 3511 GFLOP/s, about **8.44x the supplied naive kernel's throughput**, while cuBLAS is about **6.58x the tiled throughput**. NVIDIA documents shape-dependent selection among GEMM tiling strategies and larger output tiles in [Matrix Multiplication Background, section 2.3](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#typical-tile-dimensions-in-cublas-and-performance). Better reuse and a different parallelism/efficiency balance can explain part of the gap. The exact cuBLAS kernel was not profiled, and TF32 is disabled in this test.

Sources: [final clean](../raw_results/final_clean_gpu.log), [final build](../raw_results/final_build_gpu.log), [final tests](../raw_results/final_test3_gpu.log).

## Additional report calculations and coverage limits

- **Global operand float-load requests, M=N=K=1024, ignoring caches:** naive `2*M*N*K = 2,147,483,648`; tiled `2*M*N*K/TILE`. TILE=16 makes **134,217,728** requests, 16x fewer: **536,870,912 versus 8,589,934,592 bytes**. TILE=8/32 gives 268,435,456 / 67,108,864 float requests. Counts exclude bias and output stores; fewer requests do not imply the same factor of runtime improvement.
- **Encoder fc1 GELU fusion:** `1500*5120 = 7,680,000` FP32 outputs. Avoiding a separate GELU read/write pass saves `2*4*M*N = 61,440,000` logical global-memory bytes (**61.44 MB decimal; 58.59375 MiB**) and a launch. The final GEMM output store remains. No unfused timing ablation was run, so a GELU-fusion speedup cannot be quantified from these logs.
- The supplied suite names encoder q_proj, with dimensions also relevant to k/v/o, but does not independently time those layers. It has **no encoder fc2 test**. Conv2 is not a substitute for fc2. Decoder tests use M=64, rather than every sequence length between 59 and 71.
- Tests run independently of Tasks 1 and 2. Compiling their objects as part of the supplied Makefile does not mean their tests or model-level integration were run. No Task 1/2/4 changes or end-to-end inference measurements are part of this Task 3 record.

## Complete measurements for every valid configuration

The following tables preserve all 70 passing case records, including small cases, maximum errors and naive/cuBLAS comparisons. Values are copied from the original logs; no missing timings were fabricated.

<details>
<summary>Baseline: TILE=16 — 10/10 PASS</summary>

Source: [baseline_tile16.log](../raw_results/baseline_tile16.log).

| Case | Status | Max error | Time (ms) | Tiled GFLOP/s | Naive GFLOP/s | cuBLAS GFLOP/s |
|---|---|---|---|---|---|---|
| tiny 3 x 5, K = 7 | PASS | 9.76e-09 | 0.002 | 0 | 0 | 0 |
| odd sizes 37 x 100, K = 33, bias | PASS | 1.55e-07 | 0.003 | 94 | 76 | 43 |
| odd sizes, bias + GELU | PASS | 7.70e-07 | 0.005 | 197 | 132 | 155 |
| decoder q_proj (64x2048x2048) | PASS | 1.05e-05 | 0.174 | 3091 | 396 | 14113 |
| decoder gate_proj (64x6144x2048) | PASS | 1.14e-05 | 0.502 | 3209 | 407 | 15955 |
| decoder down_proj (64x2048x6144) | PASS | 3.53e-05 | 0.571 | 2822 | 386 | 17721 |
| lm_head, last token (1x59264x2048) | PASS | 1.24e-05 | 1.112 | 218 | 212 | 1134 |
| encoder q_proj (1500x1280x1280) | PASS | 9.66e-06 | 1.425 | 3450 | 414 | 20957 |
| encoder fc1 + GELU (1500x5120x1280) | PASS | 6.92e-07 | 5.608 | 3506 | 416 | 23083 |
| conv2 via im2col (1500x1280x3840) | PASS | 2.74e-05 | 4.282 | 3444 | 416 | 22575 |

</details>

<details>
<summary>TILE=8 — 10/10 PASS</summary>

Source: [tile8.log](../raw_results/tile8.log).

| Case | Status | Max error | Time (ms) | Tiled GFLOP/s | Naive GFLOP/s | cuBLAS GFLOP/s |
|---|---|---|---|---|---|---|
| tiny 3 x 5, K = 7 | PASS | 9.76e-09 | 0.002 | 0 | 0 | 0 |
| odd sizes 37 x 100, K = 33, bias | PASS | 1.55e-07 | 0.003 | 88 | 77 | 43 |
| odd sizes, bias + GELU | PASS | 7.70e-07 | 0.005 | 165 | 134 | 156 |
| decoder q_proj (64x2048x2048) | PASS | 1.05e-05 | 0.291 | 1846 | 396 | 14175 |
| decoder gate_proj (64x6144x2048) | PASS | 1.14e-05 | 0.741 | 2172 | 408 | 15920 |
| decoder down_proj (64x2048x6144) | PASS | 3.53e-05 | 0.858 | 1878 | 386 | 17740 |
| lm_head, last token (1x59264x2048) | PASS | 1.24e-05 | 0.801 | 303 | 211 | 1132 |
| encoder q_proj (1500x1280x1280) | PASS | 9.66e-06 | 2.181 | 2254 | 415 | 21017 |
| encoder fc1 + GELU (1500x5120x1280) | PASS | 6.92e-07 | 8.648 | 2273 | 416 | 23071 |
| conv2 via im2col (1500x1280x3840) | PASS | 2.74e-05 | 6.545 | 2253 | 416 | 22580 |

</details>

<details>
<summary>TILE=16, separate rebuild — 10/10 PASS</summary>

Source: [tile16.log](../raw_results/tile16.log).

| Case | Status | Max error | Time (ms) | Tiled GFLOP/s | Naive GFLOP/s | cuBLAS GFLOP/s |
|---|---|---|---|---|---|---|
| tiny 3 x 5, K = 7 | PASS | 9.76e-09 | 0.002 | 0 | 0 | 0 |
| odd sizes 37 x 100, K = 33, bias | PASS | 1.55e-07 | 0.003 | 95 | 77 | 43 |
| odd sizes, bias + GELU | PASS | 7.70e-07 | 0.005 | 197 | 133 | 156 |
| decoder q_proj (64x2048x2048) | PASS | 1.05e-05 | 0.173 | 3098 | 396 | 14161 |
| decoder gate_proj (64x6144x2048) | PASS | 1.14e-05 | 0.501 | 3215 | 408 | 15988 |
| decoder down_proj (64x2048x6144) | PASS | 3.53e-05 | 0.570 | 2826 | 386 | 17794 |
| lm_head, last token (1x59264x2048) | PASS | 1.24e-05 | 1.110 | 219 | 212 | 1132 |
| encoder q_proj (1500x1280x1280) | PASS | 9.66e-06 | 1.423 | 3455 | 415 | 20996 |
| encoder fc1 + GELU (1500x5120x1280) | PASS | 6.92e-07 | 5.600 | 3511 | 416 | 23082 |
| conv2 via im2col (1500x1280x3840) | PASS | 2.74e-05 | 4.282 | 3444 | 416 | 22582 |

</details>

<details>
<summary>TILE=32 — 10/10 PASS</summary>

Source: [tile32.log](../raw_results/tile32.log).

| Case | Status | Max error | Time (ms) | Tiled GFLOP/s | Naive GFLOP/s | cuBLAS GFLOP/s |
|---|---|---|---|---|---|---|
| tiny 3 x 5, K = 7 | PASS | 9.76e-09 | 0.003 | 0 | 0 | 0 |
| odd sizes 37 x 100, K = 33, bias | PASS | 1.55e-07 | 0.004 | 62 | 76 | 42 |
| odd sizes, bias + GELU | PASS | 7.70e-07 | 0.007 | 123 | 133 | 150 |
| decoder q_proj (64x2048x2048) | PASS | 1.05e-05 | 0.185 | 2899 | 396 | 14142 |
| decoder gate_proj (64x6144x2048) | PASS | 1.14e-05 | 0.454 | 3548 | 408 | 15960 |
| decoder down_proj (64x2048x6144) | PASS | 3.53e-05 | 0.613 | 2627 | 386 | 17736 |
| lm_head, last token (1x59264x2048) | PASS | 1.24e-05 | 1.921 | 126 | 211 | 1133 |
| encoder q_proj (1500x1280x1280) | PASS | 9.66e-06 | 1.190 | 4130 | 414 | 20941 |
| encoder fc1 + GELU (1500x5120x1280) | PASS | 6.92e-07 | 4.695 | 4188 | 416 | 23065 |
| conv2 via im2col (1500x1280x3840) | PASS | 2.74e-05 | 3.592 | 4105 | 416 | 22583 |

</details>

<details>
<summary>No shared-memory padding, TILE=16 — 10/10 PASS</summary>

Source: [no_padding.log](../raw_results/no_padding.log).

| Case | Status | Max error | Time (ms) | Tiled GFLOP/s | Naive GFLOP/s | cuBLAS GFLOP/s |
|---|---|---|---|---|---|---|
| tiny 3 x 5, K = 7 | PASS | 9.76e-09 | 0.002 | 0 | 0 | 0 |
| odd sizes 37 x 100, K = 33, bias | PASS | 1.55e-07 | 0.003 | 76 | 77 | 43 |
| odd sizes, bias + GELU | PASS | 7.70e-07 | 0.006 | 142 | 132 | 155 |
| decoder q_proj (64x2048x2048) | PASS | 1.05e-05 | 0.394 | 1363 | 396 | 14155 |
| decoder gate_proj (64x6144x2048) | PASS | 1.14e-05 | 1.138 | 1415 | 408 | 15987 |
| decoder down_proj (64x2048x6144) | PASS | 3.53e-05 | 1.317 | 1223 | 386 | 17792 |
| lm_head, last token (1x59264x2048) | PASS | 1.24e-05 | 2.643 | 92 | 211 | 1134 |
| encoder q_proj (1500x1280x1280) | PASS | 9.66e-06 | 3.354 | 1465 | 415 | 21024 |
| encoder fc1 + GELU (1500x5120x1280) | PASS | 6.92e-07 | 13.318 | 1476 | 416 | 23088 |
| conv2 via im2col (1500x1280x3840) | PASS | 2.74e-05 | 10.059 | 1466 | 416 | 22581 |

</details>

<details>
<summary>Strided W loading, TILE=16 — 10/10 PASS</summary>

Source: [strided_w.log](../raw_results/strided_w.log).

| Case | Status | Max error | Time (ms) | Tiled GFLOP/s | Naive GFLOP/s | cuBLAS GFLOP/s |
|---|---|---|---|---|---|---|
| tiny 3 x 5, K = 7 | PASS | 9.76e-09 | 0.002 | 0 | 0 | 0 |
| odd sizes 37 x 100, K = 33, bias | PASS | 1.55e-07 | 0.003 | 92 | 78 | 43 |
| odd sizes, bias + GELU | PASS | 7.70e-07 | 0.005 | 184 | 131 | 155 |
| decoder q_proj (64x2048x2048) | PASS | 1.05e-05 | 0.247 | 2177 | 396 | 14169 |
| decoder gate_proj (64x6144x2048) | PASS | 1.14e-05 | 0.713 | 2258 | 408 | 16009 |
| decoder down_proj (64x2048x6144) | PASS | 3.53e-05 | 0.877 | 1836 | 386 | 17782 |
| lm_head, last token (1x59264x2048) | PASS | 1.24e-05 | 1.617 | 150 | 211 | 1133 |
| encoder q_proj (1500x1280x1280) | PASS | 9.66e-06 | 2.069 | 2376 | 415 | 20998 |
| encoder fc1 + GELU (1500x5120x1280) | PASS | 6.92e-07 | 8.196 | 2399 | 416 | 23077 |
| conv2 via im2col (1500x1280x3840) | PASS | 2.74e-05 | 6.204 | 2377 | 416 | 22586 |

</details>

<details>
<summary>Final production, TILE=16 — 10/10 PASS</summary>

Source: [final_test3_gpu.log](../raw_results/final_test3_gpu.log).

| Case | Status | Max error | Time (ms) | Tiled GFLOP/s | Naive GFLOP/s | cuBLAS GFLOP/s |
|---|---|---|---|---|---|---|
| tiny 3 x 5, K = 7 | PASS | 9.76e-09 | 0.002 | 0 | 0 | 0 |
| odd sizes 37 x 100, K = 33, bias | PASS | 1.55e-07 | 0.003 | 94 | 77 | 42 |
| odd sizes, bias + GELU | PASS | 7.70e-07 | 0.005 | 197 | 133 | 155 |
| decoder q_proj (64x2048x2048) | PASS | 1.05e-05 | 0.174 | 3093 | 396 | 14126 |
| decoder gate_proj (64x6144x2048) | PASS | 1.14e-05 | 0.502 | 3207 | 406 | 15828 |
| decoder down_proj (64x2048x6144) | PASS | 3.53e-05 | 0.572 | 2815 | 386 | 17803 |
| lm_head, last token (1x59264x2048) | PASS | 1.24e-05 | 1.110 | 219 | 211 | 1133 |
| encoder q_proj (1500x1280x1280) | PASS | 9.66e-06 | 1.423 | 3455 | 415 | 21019 |
| encoder fc1 + GELU (1500x5120x1280) | PASS | 6.92e-07 | 5.600 | 3511 | 416 | 23087 |
| conv2 via im2col (1500x1280x3840) | PASS | 2.74e-05 | 4.282 | 3444 | 416 | 22574 |

</details>

## Raw-file index and provenance

All measurements in this report come from the unmodified cluster logs in [`raw_results/`](../raw_results/). Derived percentages and bandwidths use the displayed numbers and are rounded. The historical Mac build attempts are explicitly excluded from GPU performance/correctness results.

| Raw files | Purpose / interpretation |
|---|---|
| [environment_gpu.txt](../raw_results/environment_gpu.txt) | Actual GPU/MIG identity, CUDA compiler/driver, build revision |
| [baseline_tile16.log](../raw_results/baseline_tile16.log) | Initial reference configuration; full correctness/performance output |
| [tile8.log](../raw_results/tile8.log), [tile16.log](../raw_results/tile16.log), [tile32.log](../raw_results/tile32.log) | TILE-size ablation |
| [no_padding.log](../raw_results/no_padding.log) | Padding ablation |
| [strided_w.log](../raw_results/strided_w.log) | W-loading ablation |
| [no_second_barrier.log](../raw_results/no_second_barrier.log), [repeat2](../raw_results/no_second_barrier_repeat2.log), [repeat3](../raw_results/no_second_barrier_repeat3.log) | Three intentionally racy diagnostic runs |
| [no_second_barrier_racecheck.log](../raw_results/no_second_barrier_racecheck.log) | Architecture-specific small-case racecheck diagnostic |
| [regular status](../raw_results/no_second_barrier.exit_status), [repeat2 status](../raw_results/no_second_barrier_repeat2.exit_status), [repeat3 status](../raw_results/no_second_barrier_repeat3.exit_status), [racecheck status](../raw_results/no_second_barrier_racecheck.exit_status) | Make-wrapper statuses 2/2/2; sanitizer status 1 |
| [baseline_build.log](../raw_results/baseline_build.log), [tile8_build.log](../raw_results/tile8_build.log), [tile16_build.log](../raw_results/tile16_build.log), [tile32_build.log](../raw_results/tile32_build.log) | Compilation/linking commands; no compiler errors reported |
| [no_padding_build.log](../raw_results/no_padding_build.log), [strided_w_build.log](../raw_results/strided_w_build.log), [no_second_barrier_build.log](../raw_results/no_second_barrier_build.log) | Experimental-variant compilation/linking; no compiler errors reported |
| [final_clean_gpu.log](../raw_results/final_clean_gpu.log), [final_build_gpu.log](../raw_results/final_build_gpu.log), [final_test3_gpu.log](../raw_results/final_test3_gpu.log) | Final production clean/build/validation record |
| [environment_local.txt](../raw_results/environment_local.txt) | Earlier macOS ARM64 environment with no CUDA/GPU tools |
| [initial_build_attempt.log](../raw_results/initial_build_attempt.log), [initial_test3_attempt.log](../raw_results/initial_test3_attempt.log), [final_build_attempt.log](../raw_results/final_build_attempt.log), [final_test3_attempt.log](../raw_results/final_test3_attempt.log) | Earlier local GPU-detection failures; not cluster failures |
| [final_clean.log](../raw_results/final_clean.log) | Earlier local build-directory cleanup; not a GPU measurement |
