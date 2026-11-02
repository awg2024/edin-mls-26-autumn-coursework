// Task 2, part B: row-wise kernels built on the helpers in task2_reduce.cuh.
// Read docs/task2-reduce.md first.
//
// All three kernels work on a [rows, cols] matrix and use one block per row.
// The threads of a block walk along the row with a block-stride loop:
//
//     for (int c = threadIdx.x; c < cols; c += blockDim.x) ...
//
// so thread 0 handles columns 0, 256, 512, ..., thread 1 handles 1, 257, ...
// Neighbouring threads read neighbouring addresses, and any cols works.
#include "asr_kernels.h"
#include "task2_reduce.cuh"

constexpr int REDUCE_THREADS = 256;  // threads per block: a multiple of 32, at most 1024

// Worked example. LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias,
                                 float* y, int cols, float eps) {
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;  // this block's row
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;

  float sum = 0.0f;  // 1. mean
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += xr[c];
  const float mean = block_reduce_sum(sum) / cols;

  float sq = 0.0f;   // 2. variance
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float d = xr[c] - mean;
    sq += d * d;
  }
  const float rstd = rsqrtf(block_reduce_sum(sq) / cols + eps);  // 1 / sqrt(var + eps)

  for (int c = threadIdx.x; c < cols; c += blockDim.x)  // 3. normalise and write
    yr[c] = (xr[c] - mean) * rstd * weight[c] + bias[c];
}

// RMSNorm (text decoder): y = x / sqrt(mean(x^2) + eps) * weight
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* y, int cols,
                               float eps) {
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;

  // ======================== BLANK 2.5 =========================
  // RMSNorm of this row: y = x / sqrt(mean(x^2) + eps) * weight.

  float sq = 0.0f; // cuda private variable sq, held in register 
  for (int c = threadIdx.x; c < cols; c += blockDim.x) { // readme, threads of the block walk along the row together

      float value = xr[c]; // rms norm kernel 
      sq += value * value; } // each thread adds up sum of squared original values 

  // EXPERIMENT: shared-memory atomic sum of partial squares.
  __shared__ float shared_sq;
  if (threadIdx.x == 0) shared_sq = 0.0f;
  __syncthreads();
  atomicAdd(&shared_sq, sq);
  __syncthreads();
  float total_sq = shared_sq; // combine partial sums over warps into global var 
  float inv_rms = rsqrtf(total_sq / cols + eps); // calculate scaling factor using reciprocal

  for (int c = threadIdx.x; c < cols; c += blockDim.x) { // threads stride across the row second time to write final outut values
      yr[c] = xr[c] * inv_rms * weight[c];} // element x is multiplied by precomputed scaling factor and corresponding feature weight 
  // ============================================================
}

// Softmax over each row: y = exp(x - max) / sum(exp(x - max)).
// The textbook version reads the row three times (max, sum, write). With the
// online-softmax MaxSum from part A, the first two passes become one.
__global__ void softmax_kernel(const float* x, float* y, int cols) {
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;

  float row_max = 0.0f;  // max of the whole row
  float row_sum = 1.0f;  // sum of exp(x - row_max) over the whole row
  // ======================== BLANK 2.6 =========================
  // Pass 1: set row_max and row_sum (the sum of exp(x - row_max) over the row),
  // reading x only once, with MaxSum.
  
  MaxSum local = {-INFINITY, 0.0f};  // initialisation for thread accumulationok l

  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      MaxSum single = {xr[c], 1.0f};
      local = maxsum_merge(local, single); // call our pre-exisiting part A online softmax function 
  }

  row_max = block_reduce_max(local.m); // perform parallel reduction across all threads to find max 

  float scaled_sum = 0.0f; // init rescale local sum to global max 

  if (local.l > 0.0f) { 
      scaled_sum = local.l * expf(local.m - row_max); // calculate scaled sum with local relative to own lcocal max 
  }

  row_sum = block_reduce_sum(scaled_sum); // sums up partial rescaled exponential sums to compute final global row_sum 
  // ============================================================


  // ======================== BLANK 2.7 =========================
  // Pass 2: y = exp(x - row_max) / row_sum.
  
  for (int c = threadIdx.x; c < cols; c += blockDim.x) { // loop every thread through assigned columns 
      yr[c] = expf(xr[c] - row_max) / row_sum;} // calculate a softmax after every row_sum has been calcualted in pass 1 

  // ============================================================
}

ASR_API int launch_rmsnorm(const float* x, const float* weight, float* y, int rows,
                           int cols, float eps, cudaStream_t stream) {
  rmsnorm_kernel<<<rows, REDUCE_THREADS, 0, stream>>>(x, weight, y, cols, eps);
  RETURN_LAUNCH_ERROR();
}

ASR_API int launch_layernorm(const float* x, const float* weight, const float* bias,
                             float* y, int rows, int cols, float eps, cudaStream_t stream) {
  layernorm_kernel<<<rows, REDUCE_THREADS, 0, stream>>>(x, weight, bias, y, cols, eps);
  RETURN_LAUNCH_ERROR();
}

ASR_API int launch_softmax(const float* x, float* y, int rows, int cols,
                           cudaStream_t stream) {
  softmax_kernel<<<rows, REDUCE_THREADS, 0, stream>>>(x, y, cols);
  RETURN_LAUNCH_ERROR();
}
