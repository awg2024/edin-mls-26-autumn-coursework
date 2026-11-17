
/*
 * task 1 rope, this is an additional cuda kernel to test some hypothessi obtained from the intiial benchmakr resutls from the rope1 task. 
 *
 * Experiment A:
 * Does precomputing sine/cosine improve performance?
 *
 * Experiment B:
 * Is one fused Q/K kernel launch faster than two launches?
 *
 * This is a standalone benchmark.
 * It does NOT modify csrc/task1_rope.cu.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <utility>


// CUDA ERROR CHECKING
// Stop immediately if a CUDA operation fails.
#define CK(call) do {                                      \
    cudaError_t err = (call);                             \
    if (err != cudaSuccess) {                             \
        fprintf(stderr, "CUDA error: %s\n",                \
                cudaGetErrorString(err));                  \
        exit(1);                                          \
    }                                                     \
} while (0)



// AUDIO ENCODER TENSOR DIMENSIONS
// These are the dimensions used in the audio encoder test.

constexpr int B = 1;          // Batch size
constexpr int SEQ = 1500;     // Token positions
constexpr int HQ = 20;        // Query heads
constexpr int HK = 20;        // Key heads
constexpr int DIM = 64;       // Features per attention head
constexpr int ROT = 32;       // Number of rotated features
constexpr int HALF = ROT / 2; // 16 feature pairs per head

// Number of rows in Q and K.
constexpr int QR = B * SEQ * HQ;
constexpr int KR = B * SEQ * HK;

// Number of feature pairs needing rotation.
constexpr int QP = QR * HALF;
constexpr int KP = KR * HALF;
constexpr int TOTAL = QP + KP;

// Keep the same block size for all experiments.
// This isolates the effects of table lookup and launch fusion.
constexpr int THREADS = 256;

// Model calls RoPE 396 times per transcription.
// Benchmark 396 consecutive calls and repeat seven times.
constexpr int CYCLES = 396;
constexpr int REPEATS = 7;


// ROTATE ONE FEATURE PAIR
/*
 * This function contains the actual RoPE mathematics.
 *
 * Both the fused and split kernels call this function.
 *
 * If cost is nullptr:
 *     Compute sinf() and cosf().
 *
 * Otherwise:
 *     Read precomputed values from lookup tables.
 */

__device__ __forceinline__
void rotate(
    float* x,
    int idx,
    int heads,
    const float* inv,
    const float* cost,
    const float* sint
) {
    // Find the row and pair assigned to this thread.
    int row = idx / HALF;
    int i = idx % HALF;

    // Determine which token position this row belongs to.
    int pos = (row / heads) % SEQ;

    // Pointer to the beginning of the row.
    float* v = x + static_cast<size_t>(row) * DIM;

    // Read the original feature pair.
    float a = v[i];
    float b = v[i + HALF];

    float c;
    float s;

    if (cost != nullptr) {

        // EXPERIMENT A: Lookup table.
        // One table entry per token position and pair.
        size_t table_idx = static_cast<size_t>(pos) * HALF + i;

        c = cost[table_idx];
        s = sint[table_idx];

    } else {

        // BASELINE: Calculate trigonometry in this thread.
        float angle = pos * inv[i];

        c = cosf(angle);
        s = sinf(angle);
    }

    // Apply the same rotation in both implementations.
    v[i] = a * c - b * s;
    v[i + HALF] = b * c + a * s;
}
// fused original kernal, one launch contains q and k feature pairs (baseline to comapre) 
/*
 * One launch contains all Q and K feature pairs.
 *
 * Threads 0 ... QP-1:
 *     Process Query.
 * Threads QP ... TOTAL-1:
 *     Process Key.
 */

__global__
void fused(
    float* q,
    float* k,
    const float* inv,
    const float* cost,
    const float* sint
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= TOTAL) return;

    if (idx < QP) {

        // Query feature pair.
        rotate(q, idx, HQ, inv, cost, sint);

    } else {

        // Key feature pair.
        // Subtract QP to obtain its local index in K.
        rotate(k, idx - QP, HK, inv, cost, sint);
    }
}


// 5. SPLIT KERNEL: ONE TENSOR PER LAUNCH (q or k per launch, differing baseline we can test against)

/*
 * This kernel processes only one tensor.
 * We call it once for Q and once for K.
 * That means two kernel launches per RoPE operation.
 */

__global__
void separate(
    float* x,
    const float* inv,
    const float* cost,
    const float* sint,
    int pairs,
    int heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= pairs) return;

    rotate(x, idx, heads, inv, cost, sint);
}


// precomputed sin/cosine table to avoid replication of operations again. when we are rotating the tesnors. (e..g, in oriingal implementation it was half? so it would be agian?)
/*
 * Each thread calculates one (position, frequency) entry.
 *
 * Table shape:
 *     [SEQ, HALF]
 * We build two arrays of that shape:
 *     cost: cosine coefficients
 *     sint: sine coefficients
 * Once built, the values can be reused across Q and K.
 */

__global__
void build_table(
    const float* inv,
    float* cost,
    float* sint
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= SEQ * HALF) return;

    int pos = idx / HALF;
    int i = idx % HALF;

    float angle = pos * inv[i];

    cost[idx] = cosf(angle);
    sint[idx] = sinf(angle);
}

// choose expeirment to run launching the specifiied kernel for the additional followup benchmarks. 
/*
 * variant 0: fused_compute
 * variant 1: split_compute
 * variant 2: fused_table
 * variant 3: split_table
 *
 * Each version rotates exactly the same Q and K values.
 */

void launch(
    int variant,
    float* q,
    float* k,
    const float* inv,
    const float* c,
    const float* s
) {
    // Only variants 2 and 3 use the lookup tables.
    bool use_table = (variant == 2 || variant == 3);

    const float* pc = use_table ? c : nullptr;
    const float* ps = use_table ? s : nullptr;

    if (variant == 0 || variant == 2) {

        // ONE launch containing Q and K.
        int blocks = (TOTAL + THREADS - 1) / THREADS;

        fused<<<blocks, THREADS>>>(
            q, k, inv, pc, ps
        );

    } else {

        // TWO launches: Q first, then K.
        int q_blocks = (QP + THREADS - 1) / THREADS;
        int k_blocks = (KP + THREADS - 1) / THREADS;

        separate<<<q_blocks, THREADS>>>(
            q, inv, pc, ps, QP, HQ
        );

        separate<<<k_blocks, THREADS>>>(
            k, inv, pc, ps, KP, HK
        );
    }

    CK(cudaGetLastError());
}

//  verify numerical correctness, this is essential as initial benchmarks we saw fast maths breaking the long timestep iterations. 
/*
 * Before benchmarking, check that every implementation
 * produces the expected result for one RoPE application.
 *
 * The reference is calculated on the CPU using double
 * precision trigonometry.
 *
 * We also verify that unrotated features remain unchanged.
 */

double verify(
    int variant,
    float* dq,
    float* dk,
    const float* dinv,
    const float* dc,
    const float* ds,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& inv
) {
    // Restore original input tensors.
    CK(cudaMemcpy(
        dq, q.data(), q.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CK(cudaMemcpy(
        dk, k.data(), k.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // Run the selected implementation once.
    launch(variant, dq, dk, dinv, dc, ds);
    CK(cudaDeviceSynchronize());

    // Copy the GPU results to the CPU.
    std::vector<float> out_q(q.size());
    std::vector<float> out_k(k.size());

    CK(cudaMemcpy(
        out_q.data(), dq, out_q.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CK(cudaMemcpy(
        out_k.data(), dk, out_k.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    double max_error = 0.0;
    size_t incorrect = 0;

    // Compare either Q or K against the CPU reference.
    auto check = [&](const std::vector<float>& input,
                     const std::vector<float>& output,
                     int heads) {

        size_t rows = input.size() / DIM;

        for (size_t row = 0; row < rows; ++row) {

            int pos = static_cast<int>((row / heads) % SEQ);

            for (int i = 0; i < HALF; ++i) {

                float angle = pos * inv[i];

                double a = input[row * DIM + i];
                double b = input[row * DIM + i + HALF];

                double c = std::cos(static_cast<double>(angle));
                double s = std::sin(static_cast<double>(angle));

                double expected_a = a * c - b * s;
                double expected_b = b * c + a * s;

                // Check both rotated feature values.
                for (auto p : {
                    std::pair<int, double>(i, expected_a),
                    std::pair<int, double>(i + HALF, expected_b)
                }) {
                    double expected = p.second;
                    double got = output[row * DIM + p.first];

                    double error = std::abs(got - expected);

                    max_error = std::max(max_error, error);

                    // Absolute + relative numerical tolerance.
                    if (error > 1e-5 + 1e-4 * std::abs(expected)) {
                        incorrect++;
                    }
                }
            }

            // Unrotated dimensions must remain identical.
            for (int i = ROT; i < DIM; ++i) {
                if (input[row * DIM + i] != output[row * DIM + i]) {
                    incorrect++;
                }
            }
        }
    };

    check(q, out_q, HQ);
    check(k, out_k, HK);

    if (incorrect > 0) {
        fprintf(
            stderr,
            "Variant %d FAILED: %zu incorrect values, max error %.9g\n",
            variant, incorrect, max_error
        );
        exit(2);
    }

    return max_error;
}

// build results 
int main() {

    cudaDeviceProp gpu;
    CK(cudaGetDeviceProperties(&gpu, 0));

    fprintf(
        stderr,
        "GPU: %s, SMs: %d\n",
        gpu.name, gpu.multiProcessorCount
    );

    fprintf(
        stderr,
        "Benchmark: %d RoPE calls per trial, %d trials\n",
        CYCLES, REPEATS
    );

    // Create deterministic input tensors on the CPU.
    std::vector<float> hq(static_cast<size_t>(QR) * DIM);
    std::vector<float> hk(static_cast<size_t>(KR) * DIM);
    std::vector<float> hi(HALF);

    for (size_t i = 0; i < hq.size(); ++i) {
        hq[i] = static_cast<float>(std::sin(i * 0.013));
    }

    for (size_t i = 0; i < hk.size(); ++i) {
        hk[i] = static_cast<float>(std::cos(i * 0.017));
    }

    // Construct standard RoPE inverse frequencies.
    for (int i = 0; i < HALF; ++i) {
        hi[i] = static_cast<float>(
            1.0 / std::pow(10000.0, 2.0 * i / ROT)
        );
    }

    // GPU pointers.
    float *q, *k, *inv, *cost, *sint;

    // Allocate GPU memory.
    CK(cudaMalloc(&q, hq.size() * sizeof(float)));
    CK(cudaMalloc(&k, hk.size() * sizeof(float)));
    CK(cudaMalloc(&inv, HALF * sizeof(float)));
    CK(cudaMalloc(&cost, SEQ * HALF * sizeof(float)));
    CK(cudaMalloc(&sint, SEQ * HALF * sizeof(float)));

    // Transfer inverse frequencies to the GPU.
    CK(cudaMemcpy(
        inv, hi.data(), HALF * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // -----------------------------------------------------
    // TIME LOOKUP-TABLE CONSTRUCTION
    // -----------------------------------------------------

    cudaEvent_t setup_start, setup_end;

    CK(cudaEventCreate(&setup_start));
    CK(cudaEventCreate(&setup_end));

    CK(cudaEventRecord(setup_start));

    build_table<<<
        (SEQ * HALF + THREADS - 1) / THREADS,
        THREADS
    >>>(inv, cost, sint);

    CK(cudaGetLastError());

    CK(cudaEventRecord(setup_end));
    CK(cudaEventSynchronize(setup_end));

    float table_ms = 0.0f;

    CK(cudaEventElapsedTime(
        &table_ms, setup_start, setup_end
    ));

    fprintf(
        stderr,
        "Lookup table construction: %.6f ms\n",
        table_ms
    );

    CK(cudaEventDestroy(setup_start));
    CK(cudaEventDestroy(setup_end));

    const char* names[] = {
        "fused_compute",
        "split_compute",
        "fused_table",
        "split_table"
    };

    // CSV header: stdout is redirected to measurements.csv.
    printf(
        "variant,trial,launches_per_call,"
        "ms_per_call,effective_GBps,max_abs_error\n"
    );

    // -----------------------------------------------------
    // RUN ALL FOUR EXPERIMENTAL VARIANTS
    // -----------------------------------------------------

    for (int variant = 0; variant < 4; ++variant) {

        // Verify one application before timing.
        double error = verify(
            variant, q, k, inv, cost, sint, hq, hk, hi
        );

        fprintf(
            stderr,
            "%s correctness PASS, max error %.9g\n",
            names[variant], error
        );

        // Restore original values before benchmarking.
        CK(cudaMemcpy(
            q, hq.data(), hq.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CK(cudaMemcpy(
            k, hk.data(), hk.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        // Warm up the GPU before recording measurements.
        for (int j = 0; j < 20; ++j) {
            launch(variant, q, k, inv, cost, sint);
        }

        CK(cudaDeviceSynchronize());

        cudaEvent_t start, end;

        CK(cudaEventCreate(&start));
        CK(cudaEventCreate(&end));

        for (int trial = 1; trial <= REPEATS; ++trial) {

            // Start GPU timing.
            CK(cudaEventRecord(start));

            // Execute the equivalent of 396 RoPE calls.
            for (int j = 0; j < CYCLES; ++j) {
                launch(variant, q, k, inv, cost, sint);
            }

            // Stop GPU timing and wait for completion.
            CK(cudaEventRecord(end));
            CK(cudaEventSynchronize(end));

            float total_ms = 0.0f;

            CK(cudaEventElapsedTime(
                &total_ms, start, end
            ));

            // Mean time for one RoPE call.
            double ms_per_call = total_ms / CYCLES;

            // Nominal Q/K bytes read and written.
            // Table reads are intentionally excluded.
            double bytes =
                2.0 * (QR + KR) * ROT * sizeof(float);

            double seconds = ms_per_call * 1e-3;

            double bandwidth_GBs =
                bytes / seconds / 1e9;

            // Save this trial as one CSV row.
            printf(
                "%s,%d,%d,%.8f,%.2f,%.9g\n",
                names[variant],
                trial,
                (variant % 2) ? 2 : 1,
                ms_per_call,
                bandwidth_GBs,
                error
            );

            fflush(stdout);
        }

        CK(cudaEventDestroy(start));
        CK(cudaEventDestroy(end));
    }

    // Free all GPU memory.
    CK(cudaFree(q));
    CK(cudaFree(k));
    CK(cudaFree(inv));
    CK(cudaFree(cost));
    CK(cudaFree(sint));

    return 0;
}
