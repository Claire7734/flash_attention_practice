#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace flash_practice {

// GEMM

constexpr int constexpr_min(int a, int b) { return (a < b) ? a : b; }

#define MMA_M_FRAGMENTS_PER_ITER 2 // (MMA_M / LDMATRIX_MAT_SIZE)
#define MMA_N_FRAGMENTS_PER_ITER 1 // (MMA_N / LDMATRIX_MAT_SIZE)
#define MMA_K_FRAGMENTS_PER_ITER 2 // (MMA_K / LDMATRIX_MAT_SIZE)

// It's possible for K_fragments_A != K_fragments_B because either tensor can be buffered over sub-tiles.
template <typename value_t, const int M_fragments, const int N_fragments,
          const int K_fragments_A, const int K_fragments_B,
          typename accum_t = float>
__forceinline__ __device__ constexpr void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments_A],
    uint32_t (&regs_B)[N_fragments][K_fragments_B],
    accum_t (&regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT],
    int A_col_fragment_offset = 0, int B_col_fragment_offset = 0) {
    constexpr int K_iters = constexpr_min(K_fragments_A, K_fragments_B);
    #pragma unroll
    for (int k = 0; k < K_iters; k += MMA_K_FRAGMENTS_PER_ITER) {
        #pragma unroll
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            #pragma unroll
            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1],
                    
                    regs_A[m][k + A_col_fragment_offset],
                    regs_A[m + 1][k + A_col_fragment_offset],
                    regs_A[m][k + 1 + A_col_fragment_offset],
                    regs_A[m + 1][k + 1 + A_col_fragment_offset],
                    
                    regs_B[n][k + B_col_fragment_offset],
                    regs_B[n][k + 1 + B_col_fragment_offset],
                    
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}

// Softmax

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
scale_S_accum(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
              const accum_t &softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        FA_UNROLL
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] *= softmax_scale;
        }
    }
}

// This mask indicates that every thread in the warp participates in the shuffle
#define SHFL_ENTIRE_WARP_MASK 0xffffffff
 
template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void
calc_row_max(
	accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m_next)[QO_fragments],
    accum_t (&m_cur)[QO_fragments]
) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        m_next[q] = m_cur[q];
 
        // Calculate max for row across all in-thread registers.
        #pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            m_next[q] = max(m_next[q], S_accum[q][k]);
        }
 
        // Group reduction
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 2),
                        m_next[q]);
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 1),
                        m_next[q]);
    }
}

template <bool optimized_softmax, int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void
scale_l_O(
	accum_t (&m_next)[QO_fragments],
	accum_t (&m_cur)[QO_fragments],
    accum_t (&l)[QO_fragments],
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
    accum_t softmax_scale
) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t scale;
        if constexpr (optimized_softmax) {
            scale = exp2f((m_cur[q] - m_next[q]) * softmax_scale);
        } else {
            scale = expf(m_cur[q] - m_next[q]);
        }

        m_cur[q] = m_next[q];
        l[q] *= scale;
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] *= scale;
        }
    }
}

template <bool optimized_softmax, int QO_fragments, int KV_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void
exponentiate_tensor(
	accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m)[QO_fragments],
    accum_t softmax_scale
) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t max_scaled;
        if constexpr (optimized_softmax) {
            max_scaled = m[q] * softmax_scale;
        }
        #pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            if constexpr (optimized_softmax) {
                S_accum[q][k] =
                    exp2f(S_accum[q][k] * softmax_scale - max_scaled);
            } else {
                S_accum[q][k] = expf(S_accum[q][k] - m[q]);
            }
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void
update_row_exp_sum(
	accum_t (&P_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]
) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            l[q] += P_accum[q][d_head];
        }
    }
}

} // flash_practice
