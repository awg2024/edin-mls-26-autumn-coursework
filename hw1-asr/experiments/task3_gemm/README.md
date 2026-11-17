# Task 3 tiled GEMM experiment record

The completed cluster analysis is in **[results/RESULTS.md](results/RESULTS.md)**,
with all configuration results, comparisons and interpretation. Original logs
are now in [raw_results/](raw_results/). The workflow record below predates their
import; the completed report supersedes its pending-results notes.

## Status and environment

Updated on 2026-10-27 with user-supplied university-cluster console output.
The final production run reports **all ten Task 3 cases passing**. Its measured
results are recorded below. The intentionally missing-second-barrier diagnostic
reports races and an incorrect result, as expected. Detailed ablation logs have
not yet been supplied to this local checkout, so their performance conclusions
remain pending.

| Item | Cluster environment / evidence |
|---|---|
| GPU | NVIDIA H200 MIG 3g.71gb, sm_90, 60 SMs |
| Reported approximate peak memory bandwidth | 2407 GB/s |
| CUDA installation | Build links `/opt/cuda-12.8.0/lib64`; full `nvcc --version` output pending |
| Build flags shown | `-O3 -std=c++17 -lineinfo -diag-suppress 177 -arch=sm_90 -Xcompiler -fPIC` |
| Race checker | Available on cluster; diagnostic transcript supplied |
| Final source configuration | TILE=16, `ws[16][17]`, coalesced W loads, both barriers |

[results/final_test3_gpu_user_supplied.log](results/final_test3_gpu_user_supplied.log)
is a verbatim transcription of the final `make test3` output shared by the user,
not a locally executed benchmark. The original `final_test3_gpu.log`, build logs,
environment capture, and ablation logs remain on the cluster. Import those files
to preserve the complete original experiment record. Do not relabel this final
run as the earlier `baseline_tile16.log` measurement.

Earlier local checks on macOS 26.5.1 ARM64 had no CUDA/GPU tools: `make` and
`make test3` exited 2 at GPU detection, and `make clean` exited 0. Those attempts
are not GPU measurements; the cluster's successful run supersedes their
correctness limitation. No additional local CUDA execution was attempted.

Tasks 1, 2, and 4 and all supplied tests are unchanged. Task 3 has no correctness
dependency on Tasks 1 or 2. `make` compiles their supplied source files as required
by the existing Makefile, but this procedure runs only the Task 3 tests. A separate
model-level integration check can wait until Tasks 1 and 2 are complete.

## Methodology and coverage

Use the unchanged `tests/test_task3_gemm.cu` via `make test3` for every configuration.
It checks three small shapes against a double-precision CPU reference and seven
larger shapes against cuBLAS, then applies the reference bias/GELU. The tolerance
is `abs(error) <= 1e-3 + 1e-3 * abs(reference)`; nonfinite outputs fail.

The supplied timing helper uses CUDA events, three warmups and twenty timed
launches for the tiled kernel and cuBLAS, and one warmup and five timed launches
for the naive kernel. Preserve those settings and record complete stdout/stderr
and command exit status. Keep GPU allocation, compiler flags, and clocks/load
conditions consistent between configurations. Capture GPU/MIG identity, CUDA
compiler and driver versions; repeat measurements if run-to-run variation could
change a conclusion. Compare the same cases across configurations.

Reported throughput is `2*M*N*K / (seconds * 1e9)` GFLOP/s and excludes epilogue
FLOPs. The tiled timing includes requested bias/GELU, whereas the naive and cuBLAS
comparison timings are GEMM only. cuBLAS uses `CUBLAS_PEDANTIC_MATH` (plain FP32,
no TF32). Do not attribute a gap here to reduced-precision Tensor Core execution.
The supplied test prints comparison throughput only for passing cases; a failing
or racy configuration's timings are not evidence of a valid speedup.

The following measurements come from the **final production run**, not the
initial baseline run. All three small cases also passed, with maximum errors
9.76e-09, 1.55e-07, and 7.70e-07 respectively. The maximum error over all ten
cases was 3.53e-05.

| Supplied case (M,N,K) | Correct? | Max error | Tiled ms | Tiled GFLOP/s | Naive GFLOP/s | cuBLAS GFLOP/s |
|---|---|---|---|---|---|---|
| decoder q_proj (64,2048,2048) | PASS | 1.05e-05 | 0.174 | 3093 | 396 | 14126 |
| decoder gate_proj (64,6144,2048) | PASS | 1.14e-05 | 0.502 | 3207 | 406 | 15828 |
| decoder down_proj (64,2048,6144) | PASS | 3.53e-05 | 0.572 | 2815 | 386 | 17803 |
| lm_head (1,59264,2048) | PASS | 1.24e-05 | 1.110 | 219 | 211 | 1133 |
| encoder q_proj (1500,1280,1280) | PASS | 9.66e-06 | 1.423 | 3455 | 415 | 21019 |
| encoder fc1 + GELU (1500,5120,1280) | PASS | 6.92e-07 | 5.600 | 3511 | 416 | 23087 |
| conv2 via im2col (1500,1280,3840) | PASS | 2.74e-05 | 4.282 | 3444 | 416 | 22574 |

The authoritative suite has no encoder fc2 (1500,1280,5120) case. It names only
encoder q_proj, whose dimensions also represent k/v/o projections, and decoder
q/gate/down, whose dimensions represent o/up as applicable. No separate k/v/o,
encoder fc2, or decoder sequence-length 59/71 measurements are claimed. Do not
relabel conv2 as encoder fc2. No replacement tests have been added.

## Experiments and report table

Unless specified, every variant uses TILE=16, padding, coalesced W loads, both
barriers, and the same fused epilogue. Only the named setting changes. The user
has shown that experiment logs were saved on the cluster, but individual ablation
measurements are not yet available here. Pending means results not supplied,
rather than a claim that the experiment was not executed.

| Experiment | Configuration | Representative case | Time | GFLOP/s or GB/s | Correct? | Interpretation/status |
|---|---|---|---|---|---|---|
| Baseline | TILE=16, padded, coalesced | encoder fc1 | — | — | Pending | Original baseline log needed |
| A | TILE=8 | encoder fc1 | — | — | Pending | Encoder throughput winner undetermined |
| A | TILE=16, separate rebuild | encoder fc1 | — | — | Pending | Check agreement with baseline |
| A | TILE=32 | encoder fc1 / lm_head | — | — | Pending | Only 1/32 thread rows useful for lm_head |
| B | `ws[TILE][TILE]` | encoder fc1 | — | — | Pending | Padding comparison needs its log |
| C | Strided W loads | lm_head / encoder fc1 | — | — | Pending | Coalescing comparison needs its log |
| D | Omit second barrier, diagnostic only | small GELU case | — | — | FAIL | Racecheck: 2 errors, 2 warnings |
| Final production | TILE=16, padded, coalesced | encoder fc1 | 5.600 ms | 3511 GFLOP/s | PASS | About 8.44x naive throughput |
| E | Final production tiled | lm_head | 1.110 ms | ~437.4 GB/s | PASS | Effective bandwidth, weight bytes only |
| E | Final run cuBLAS comparison | lm_head | ~0.21425 ms | ~2266 GB/s | PASS reference | Time inferred from rounded GFLOP/s |

For A, select a winner from all large encoder cases, especially q_proj and fc1,
and report any disagreement between shapes. A larger tile increases reuse but
also changes thread count, shared memory, and scheduling/resource requirements.
At M=1 and TILE=32, exactly one of 32 thread rows (32 of 1024 threads, 3.125%) has
valid output coordinates; the remaining 31 still load tiles, compute, and reach
barriers. This establishes wasted arithmetic, not a measured latency ranking.

For B, compare `no_padding.log` with `baseline_tile16.log` at TILE=16. Report both
latency ratio `t_unpadded/t_padded` and throughput loss
`100*(1 - GF_unpadded/GF_padded)` percent. Removing padding makes the column-wise
`ws[tx][q]` accesses use a stride of 16 floats, mapping distinct rows to just two
of 32 banks. Padding changes the stride to 17. Bank mapping explains the
hypothesis; the loss on the allocated GPU remains unknown.

For C, compare `strided_w.log` with the same baseline, including encoder and
lm_head cases. The temporary version below swaps loader responsibilities and
shared-memory store indices together, preserving `ws[n][k]` and the dot product.
Adjacent `tx` lanes then read different W rows, separated by K floats, rather than
adjacent K elements of one row. More memory transactions are expected; magnitude
and any exceptions require actual measurements.

For D, record failing case names from each full-suite run, repeatability of
failures, and racecheck hazard messages/counts in addition to exit status. A
regular test pass alone cannot establish absence of a shared-memory race.

The user-supplied racecheck transcript reports **4 hazards displayed (2 errors,
2 warnings)**, involving tile writes at source lines 36/46 and reads in the
multiply loop at line 55. Under racecheck, the tiny case and the K=33 biased case
pass; the K=129 bias+GELU case fails with maximum error 1.15e-01 and 36 incorrect
values out of 3465. A preceding regular diagnostic run also reports SOME CASES
FAILED, but its individual failing case lines were not provided. Repeated-run
determinism and production-kernel racecheck results are not established by the
available excerpts. Import the complete diagnostic logs for those details.

## Commands for the teaching cluster

Transfer this implementation to the teaching-cluster checkout first. From its
head node, request a GPU shell following the repository's setup instructions:

```bash
srun -p Teaching --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
cd ~/edin-mls-26-autumn-coursework/hw1-asr  # adjust to your checkout location
```

The standalone CUDA tests do not require model downloads or an end-to-end run.
Inside that GPU shell, use Bash and capture the environment and baseline:

```bash
set -euo pipefail
task3_results="$PWD/experiments/task3_gemm/results"
mkdir -p "$task3_results"
{
  date -u
  uname -sm
  nvidia-smi
  nvidia-smi -L
  nvcc --version
  command -v compute-sanitizer || true
  git rev-parse HEAD
  git diff -- csrc/task3_gemm.cu
} > "$task3_results/environment_gpu.txt" 2>&1

make 2>&1 | tee "$task3_results/baseline_build.log"
make test3 2>&1 | tee "$task3_results/baseline_tile16.log"
```

The following creates temporary source copies and independent build directories
using the Makefile's existing `SRC` and `BUILD` overrides. The production source
is never changed by these experiments. No supplied test is changed or replaced.
Python 3 is used only to prepare variants, not as a correctness test.

```bash
task3_variant_root=$(mktemp -d)
trap 'rm -rf -- "$task3_variant_root"' EXIT
task3_sm=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.[:space:]')
export TASK3_VARIANT_ROOT="$task3_variant_root"
python3 - <<'PY'
import os
from pathlib import Path
import shutil

root = Path(os.environ['TASK3_VARIANT_ROOT'])
baseline = Path('csrc/task3_gemm.cu').read_text()

def replace_once(source, old, new):
    assert source.count(old) == 1, 'Baseline changed; review variant transformation'
    return source.replace(old, new, 1)

strided = '''    const int w_row = blockIdx.x * TILE + tx;
    const int w_k = k0 + ty;
    ws[tx][ty] = (w_row < N && w_k < K)
                     ? w[static_cast<size_t>(w_row) * K + w_k]
                     : 0.0f;'''
coalesced = '''    const int w_row = blockIdx.x * TILE + ty;
    ws[ty][tx] = (w_row < N && k < K)
                     ? w[static_cast<size_t>(w_row) * K + k]
                     : 0.0f;'''
variants = {
    'tile8': replace_once(baseline, 'constexpr int TILE = 16;', 'constexpr int TILE = 8;'),
    'tile16': baseline,
    'tile32': replace_once(baseline, 'constexpr int TILE = 16;', 'constexpr int TILE = 32;'),
    'no_padding': replace_once(baseline, 'float ws[TILE][TILE + 1];', 'float ws[TILE][TILE];'),
    'strided_w': replace_once(baseline, coalesced, strided),
    'no_second_barrier': replace_once(
        baseline,
        '    __syncthreads();  // Finish all reads before the next tile overwrites them.\n',
        ''),
}
for name, source in variants.items():
    destination = root / name / 'csrc'
    shutil.copytree('csrc', destination)
    (destination / 'task3_gemm.cu').write_text(source)
PY

run_task3_variant() {
  local name="$1"
  local build="$task3_variant_root/$name/build/sm_$task3_sm"
  make SM="$task3_sm" SRC="$task3_variant_root/$name/csrc" BUILD="$build" \
    2>&1 | tee "$task3_results/${name}_build.log" || return $?
  make test3 SM="$task3_sm" SRC="$task3_variant_root/$name/csrc" BUILD="$build" \
    2>&1 | tee "$task3_results/$name.log"
}

for task3_variant in tile8 tile16 tile32 no_padding strided_w; do
  run_task3_variant "$task3_variant"
done

# The diagnostic may fail. Preserve its exit status and continue to racecheck.
if run_task3_variant no_second_barrier; then
  printf '0\n' > "$task3_results/no_second_barrier.exit_status"
else
  printf '%s\n' "$?" > "$task3_results/no_second_barrier.exit_status"
fi
for task3_repeat in 2 3; do
  if make test3 SM="$task3_sm" \
      SRC="$task3_variant_root/no_second_barrier/csrc" \
      BUILD="$task3_variant_root/no_second_barrier/build/sm_$task3_sm" \
      2>&1 | tee "$task3_results/no_second_barrier_repeat${task3_repeat}.log"; then
    printf '0\n' > "$task3_results/no_second_barrier_repeat${task3_repeat}.exit_status"
  else
    printf '%s\n' "$?" > "$task3_results/no_second_barrier_repeat${task3_repeat}.exit_status"
  fi
done

if command -v compute-sanitizer >/dev/null 2>&1; then
  if compute-sanitizer --tool racecheck --error-exitcode=1 \
      "$task3_variant_root/no_second_barrier/build/sm_$task3_sm/test_task3_gemm" small \
      2>&1 | tee "$task3_results/no_second_barrier_racecheck.log"; then
    printf '0\n' > "$task3_results/no_second_barrier_racecheck.exit_status"
  else
    printf '%s\n' "$?" > "$task3_results/no_second_barrier_racecheck.exit_status"
  fi
else
  printf 'compute-sanitizer unavailable in PATH\n' > "$task3_results/racecheck_unavailable.txt"
fi
```

The sanitizer executable above is the architecture-specific
`build/sm_XX/test_task3_gemm small` from the intentionally racy temporary copy.
Three runs can expose variation but cannot prove determinism. Inspect logs before
recording which cases fail. Never copy the racy variant back into `csrc/`.

Finish with the unchanged production source and a clean build:

```bash
make clean 2>&1 | tee "$task3_results/final_clean_gpu.log"
make 2>&1 | tee "$task3_results/final_build_gpu.log"
make test3 2>&1 | tee "$task3_results/final_test3_gpu.log"
git diff --check
```

Success requires the final log to end with `Task 3 (tiled GEMM): ALL PASS` and all
ten cases to pass. Transcribe actual numbers into the tables and state the GPU
and CUDA versions from `environment_gpu.txt`. If a different TILE is eventually
retained, record the measured reason and repeat the final clean build/test.

## lm_head effective bandwidth

For M=1, N=59264, K=2048 with FP32 weights:

```text
weight bytes = N*K*4 = 485,490,688 bytes
             = 485.490688 MB = 0.485490688 GB (decimal)
             = 463 MiB
effective bandwidth [GB/s] = 485.490688 / time_ms
```

This estimates one complete weight read per invocation, excludes the much smaller
X/output traffic (8,192 input bytes and 237,056 output bytes), and is effective
bandwidth rather than measured DRAM traffic. Cache reuse across timed launches,
transaction overhead, and redundant transfers can change actual DRAM traffic.

The test prints tiled time directly but prints cuBLAS throughput, not its time.
Recover approximate cuBLAS time from its rounded GFLOP/s value G:

```text
FLOPs = 2*M*N*K = 242,745,344
cuBLAS time_ms ~= FLOPs / (G * 1e6) = 242.745344 / G
cuBLAS effective bandwidth ~= 485.490688 / time_ms = 2*G GB/s
```

For the supplied final production run:

| Implementation | Timing basis | Approximate time | Effective bandwidth |
|---|---|---:|---:|
| Tiled | Printed kernel time | 1.110 ms | 437.4 GB/s |
| cuBLAS | Printed 1133 GFLOP/s | 0.21425 ms | 2266 GB/s |

These values use decimal GB/s. cuBLAS time is inferred from rounded throughput;
the tiled time is also rounded by the test. These are final-run measurements,
not initial-baseline measurements. Do not substitute the naive kernel's
throughput or the GPU's advertised bandwidth for a timing.

## Analytical report answers

1. **Global float-load requests, ignoring caches.** For M=N=K=1024, the supplied
   naive kernel loads two floats per inner-loop iteration per output:
   `2*M*N*K = 2,147,483,648` float requests. A TILE=T tiled kernel, with dimensions
   divisible by T, loads `2*T*T` floats per block per K tile, for
   `(M/T)*(N/T)*(K/T)` tiles: `2*M*N*K/T` requests. At T=16 this is **134,217,728**
   float requests, 16 times fewer (536,870,912 operand bytes versus
   8,589,934,592). At T=8/32 it is 268,435,456 / 67,108,864 requests. These counts
   exclude bias, output stores, shared-memory reads, and cache/transaction effects.
   They are not a prediction of a 16-fold speedup.

2. **A cuBLAS optimisation.** cuBLAS selects among GEMM tiling strategies using
   shape-dependent heuristics, including substantially larger output tiles than
   this fixed 16x16 exercise. Increased reuse and a better balance of parallelism
   and tile efficiency can explain part of a performance gap. NVIDIA documents
   this in [Matrix Multiplication Background, section 2.3](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#typical-tile-dimensions-in-cublas-and-performance).
   This is an optimisation explanation, not a claim about the exact implementation
   selected for an unexecuted case. The test disables TF32, so TF32 acceleration
   is not an explanation for its cuBLAS comparison.

3. **GELU fusion traffic saved.** Encoder fc1 produces `1500*5120 = 7,680,000`
   FP32 elements. A separate in-place GELU reads and rewrites each element, adding
   `2*4*M*N = 61,440,000` bytes (61.44 MB decimal, 58.59375 MiB) of logical global
   traffic. Fusion eliminates that extra read/write pass and one kernel launch;
   the final GEMM output store remains. This is an analytical traffic saving,
   not a measured speedup or necessarily the same number of DRAM bytes.

4. **Both barriers and edge threads.** The first barrier ensures every shared
   tile write completes and is visible before any warp consumes the tile. The
   second ensures all warps finish consuming the current tile before a faster
   warp overwrites it for the next K iteration. Without it, writes of tile i+1
   can race with reads of tile i. Out-of-range output threads can still load X
   or W values needed by valid outputs; all threads must participate in the
   block-wide barriers. Each thread loads a valid value or zero, performs the
   shared-memory loop, and only guards the final bias/activation/store. No early
   return or divergent barrier is used. The diagnostic racecheck transcript
   confirms read/write hazards when the second barrier is removed, as recorded
   above; production racecheck results have not been supplied.

## Final selected configuration

TILE=16, `ws[TILE][TILE+1]`, coalesced `[N,K]` W reads, both `__syncthreads()`
barriers, zero-filled edge tiles, guarded optional bias, and optional fused GELU.
This retains the requested safe baseline; no measurements establish a better
setting. No intentionally degraded variant is present in the production kernel.
The supplied final cluster run reports all ten correctness cases passing.
Choosing a different TILE and quantifying padding/coalescing effects still
requires importing and comparing the individual ablation logs.
