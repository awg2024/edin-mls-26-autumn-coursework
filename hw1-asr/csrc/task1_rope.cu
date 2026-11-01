// Task 1: rotary position embedding (RoPE). Read docs/task1-rope.md first.
//
// Before attention, every query and key vector is rotated by an angle that depends
// on the token's position, so attention can tell tokens apart by where they are.
// One vector has head_dim numbers. RoPE pairs element i with element i + half,
// for i = 0 .. half-1 (half = rotary_dim / 2), and rotates each pair:
//
//     angle  = pos * inv_freq[i]
//     new_a  = a * cos(angle) - b * sin(angle)       a = x[i]
//     new_b  = b * cos(angle) + a * sin(angle)       b = x[i + half]
//
// Elements rotary_dim .. head_dim-1 are left unchanged (the audio encoder only
// rotates the first half of each head; the text decoder rotates all of it).
//
// Memory layout, straight out of the q/k projections (no transposes):
//     q: [batch, seq_len, num_q_heads, head_dim]
//     k: [batch, seq_len, num_k_heads, head_dim]
// Seen as a 2D array, each row is one head of one token.
//
// One launch rotates q AND k: one thread per pair, q's pairs first, then k's.
#include "asr_kernels.h"
#include "common.cuh"

constexpr int ROPE_THREADS = 256; // each block contains 256 threads 

__global__ void rope_kernel(
  float* q,  // pointer for memory query numbers begin 
  float* k,  // pointer for memory key numbers being 
  const float* inv_freq, // pointer for memory precomputed inverse freq  
  int seq_len,  // receives query and key tensors and rotates features in 2d
  int num_q_heads, 
  int num_k_heads,
  int q_rows,
  int k_rows,
  int head_dim, 
  int rotary_dim) { 

  const int half = rotary_dim / 2; // number of pairs in a single vector 
  const int q_pairs = q_rows * half;
  const int k_pairs = k_rows * half;

  int idx = 0;  // this thread's pair number, counting q's pairs first, then k's
  
  // ======================== BLANK 1.1 =========================
  // Set idx to this thread's global index. Threads without a pair
  // (q_pairs + k_pairs in total) do nothing.
  
  // this global thread index tells each thread its unique job number inside each block
  idx = blockIdx.x * blockDim.x + threadIdx.x;  

  // threads without pair do nothing to avoid OOM error 
  if (idx >= (q_pairs + k_pairs)) return;
  // ============================================================

  float* x = q;   // the tensor this thread works on, starts on q key 
  int num_heads = num_q_heads;

  // ======================== BLANK 1.2 =========================
  // Threads past q's pairs work on k: make x, num_heads and idx describe the
  // tensor this thread works on, with idx counting from 0 inside it.

  if (idx >= q_pairs) { // if idx is greater than q-pairs
    x = k; // change to pair k processing 
    num_heads = num_k_heads;
    idx -= q_pairs; // local pair index within key, return to start 
  }

  // ============================================================

  int row = 0;  // which row (one head of one token) of x
  int i = 0;    // which pair inside that row, 0 .. half-1
  int pos = 0;  // position of the token this row belongs to
  
  // ======================== BLANK 1.3 =========================
  // Set row, i (the pair inside the row) and pos (the token position of the row).

  row = idx / half;  // what head vector is the thread in  
  i = idx % half; //  whcih feautre pairs is within that row
  pos = (row / num_heads) % seq_len; // what is the token position that this row belongs to
  // ============================================================

  float* v = x + static_cast<size_t>(row) * head_dim;  // start of this vecotr row

  // ======================== BLANK 1.4 =========================
  // Rotate the pair (v[i], v[i + half]) in place.

  float a = v[i]; // reads first component pair 
  float b = v[i + half]; // reads second half of vector 

  // original implementation; 
  float angle = pos * inv_freq[i]; // calculate rotation given position and frequency 
  float c = cosf(angle); // calls fast CUDA cos & sin functions 
  float s = sinf(angle); 

  // optimised precompute arrays of rotational angles 
  //int table_idx = pos * half + i;
  //float c = cos_table[table_idx];
  //float s = sin_table[table_idx];

  v[i] = a * c - b * s; // computes new rotational angle given vector position
  v[i + half] = b * c + a * s;
  // ============================================================


}

ASR_API int launch_rope(float* q, float* k, const float* inv_freq, int batch, int seq_len,
                        int num_q_heads, int num_k_heads, int head_dim, int rotary_dim,
                        cudaStream_t stream) {
  int q_rows = batch * seq_len * num_q_heads;
  int k_rows = batch * seq_len * num_k_heads;
  int total_pairs = (q_rows + k_rows) * (rotary_dim / 2);
  int blocks = ceil_div(total_pairs, ROPE_THREADS);
  
  // one q/k launch 
  rope_kernel<<<blocks, ROPE_THREADS, 0, stream>>>(q, k, inv_freq, seq_len, num_q_heads,
                                                    num_k_heads, q_rows, k_rows, head_dim,
                                                    rotary_dim);

  RETURN_LAUNCH_ERROR();
}
