#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "common.h"
#include "ptx_functions.cuh"
#include "utils.h"

namespace flash {

// Dimensions of the mma instruction we're using
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define MMA_M_FRAGMENTS_PER_ITER 2 // (MMA_M / LDMATRIX_MAT_SIZE)
#define MMA_N_FRAGMENTS_PER_ITER 1 // (MMA_N / LDMATRIX_MAT_SIZE)
#define MMA_K_FRAGMENTS_PER_ITER 2 // (MMA_K / LDMATRIX_MAT_SIZE)

template <typename _A_t, typename _B_t, typename _C_t, int total_K_fragments,
          int load_K_fragments_per_iter, typename value_t_>
struct GEMM {
    using A_t = _A_t;
    using B_t = _B_t;
    using C_t = _C_t;
    using value_t = value_t_;

    static constexpr int TotalKTiles = total_K_fragments;
    static constexpr int LoadKTilesPerIter = load_K_fragments_per_iter;

    static constexpr bool DoubleBufferA =
        !A_t::load_entire_block_into_rf && A_t::mma_load_stages > 1;
    static constexpr bool DoubleBufferB =
        !B_t::load_entire_block_into_rf && B_t::mma_load_stages > 1;
    static constexpr bool DoubleBuffer = DoubleBufferA || DoubleBufferB;
};

// split along K dimension
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

template <typename GEMM>
__device__ constexpr void matmul(typename GEMM::A_t &A, typename GEMM::B_t &B,
                                 typename GEMM::C_t &C) {
    // If ::load_entire_block_into_rf is set for either A_t or B_t, then
    // we assume the block has already been loaded.
    using A_t = typename GEMM::A_t; // Q or P
    using B_t = typename GEMM::B_t; // K or V

    constexpr int fragments_per_iter = 2;

    // GEMM::TotalKFragments is
    // - d_head / 8 for QK^T
    // - B_c / 8    for PV
    #pragma unroll
    for (int k = 0; k < GEMM::TotalKFragments; k += fragments_per_iter) {
		// Load fragments along K dimension (2 at a time)
		// Q is pre-loaded, P is computed in RF - only load if needed
		if constexpr (!A_t::load_entire_block_into_rf) {
			A.copy_SM2RF(k);  // Load Q fragments from SMEM
		}
		// Always load K/V fragments from SMEM (2 fragments per iteration)
		B.copy_SM2RF(k);

		// Calculate column offsets for accessing the right fragment data
        int A_col_offset = A_t::load_entire_block_into_rf ? k : 0;
        int B_col_offset = B_t::load_entire_block_into_rf ? k : 0;

        // Perform outer product: each A row × each B column
        // This gives optimal fragment reuse compared to row-by-row approach
        warp_fragment_mma_f32_accum(A.data(), B.data(), C.data(),
                                    A_col_offset, B_col_offset);
    }
}
}