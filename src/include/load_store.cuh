#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common.h"
#include "ptx_functions.cuh"
#include "swizzling.cuh"

namespace flash {

struct LDSTCommon {
    const bool swizzled;
    const bool async_copy;
};

struct TileLayout {
    const int row_fragments;
    const int col_fragments;
};

// constexpr non-type template parameter containing parameters for LD/ST for a
// block (Q, K, V, or O) from GMEM to SMEM and vice versa, and also loading from
// SMEM to the RF.
struct TensorLDSTConfig {
    // Tile layout for shared memory and RF.
    const TileLayout GSM;
    const TileLayout RF;

    const LDSTCommon Common;
 
	// Block specific properties
    const bool transposed;
    const int block_size;
    const int smem_cols;
 
    // # of rows a warp in a thread-block independently loads/stores. it is equivalent to GSM.row_fragments * 8.
    const int warp_ldst_rows;
    // Whether not the warp will compute over the entire block.
    // This is false for (Q&O&S) and true for (K&V).
    const bool compute_over_entire_block;

    const bool load_entire_block_into_rf;
    const int mma_load_stages;
};

template <typename T>
struct GM2SM_async {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        cp_async<BYTES_PER_VEC4_ACCESS>(smem, gmem);
    }
};

template <typename T>
struct GM2SM {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(smem)[0] = reinterpret_cast<uint4 *>(gmem)[0];
    }
};

template <typename T>
struct SM2GM {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(gmem)[0] = reinterpret_cast<uint4 *>(smem)[0];
    }
};
 
template <typename op, /* either GM2SM_async or SM2GM */
          TensorLDSTConfig CFG,
          typename value_t,
          typename index_t = int64_t>
__forceinline__ __device__ constexpr void copy_block_GSM(
	value_t *gmem,
	value_t *smem,
    index_t gmem_seq_stride,
    const int lane_id) {
    constexpr int n_row_iters =
        CFG.GSM.row_fragments * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;
 
    constexpr int col_fragments_per_iter = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    constexpr int col_fragments_per_row = CFG.smem_cols / COLS_PER_FRAGMENT;
 
    const int thread_row = lane_id / col_fragments_per_iter;
    const int thread_col_fragment = lane_id % col_fragments_per_iter;
 
    #pragma unroll
    for (int r = 0; r < n_row_iters; ++r) {
        const int cur_row = r * GSM_LDST_ROWS_PER_ITER + thread_row;
        #pragma unroll
        for (int c = 0; c < col_fragments_per_row;
             c += col_fragments_per_iter) {
            const int gmem_col_fragment = c + thread_col_fragment;
            // Apply swizzling to prevent bank conflicts during later column-wise access in `ldmatrix`.
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments_per_row, CFG.Common.swizzled>(
                                    cur_row,gmem_col_fragment);
 
            op()(&gmem[cur_row * gmem_seq_stride +
                       gmem_col_fragment * COLS_PER_FRAGMENT],
                 &smem[cur_row * CFG.smem_cols +
                       smem_col_fragment * COLS_PER_FRAGMENT]);
        }
    }
}

// Q and Kj
 
template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id,
    const int col_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;
 
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;
 
    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;
 
    #pragma unroll
    for (int r = 0; r < CFG.RF.row_fragments; r += row_fragments_per_iter) {
        const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
        #pragma unroll
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            // Use swizzled addresses to match the layout from GMEM→SMEM transfers
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(
                                    cur_row, thread_col_fragment + c + col_fragment_offset);
 
            ldmatrix_x4(&smem[cur_row * CFG.smem_cols +
                        smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                        regs[r][c], regs[r + 1][c], regs[r][c + 1],
                        regs[r + 1][c + 1]);
        }
    }
}

// Vj
template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id,
    const int row_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;
 
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;
 
    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;
 
    #pragma unroll
    for (int r = 0; r < CFG.RF.col_fragments; r += row_fragments_per_iter) {
        const int cur_row =
            thread_row + (r + row_fragment_offset) * ROWS_PER_FRAGMENT;
        #pragma unroll
        for (int c = 0; c < CFG.RF.row_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = 
                get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(
                    cur_row, thread_col_fragment + c);
 
            ldmatrix_x4_transpose(
                &smem[cur_row * CFG.smem_cols +
                      smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                regs[c][r], regs[c][r + 1], regs[c + 1][r], regs[c + 1][r + 1]);
        }
    }
}

template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_RF2SM(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id
) {
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT;
    constexpr int col_fragments_per_iter = 1;
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
 
    constexpr int elems_per_store = 2;
    const int thread_row = lane_id / 4;
    const int thread_inner_col = (lane_id % 4) * elems_per_store;
 
    #pragma unroll
    for (int r = 0; r < CFG.RF.row_fragments; ++r) {
        const int cur_row = thread_row + r * rows_per_iter;
        #pragma unroll
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            // const int smem_col_fragment = c;
            // // Apply swizzling to maintain consistent layout for later SMEM→GMEM transfers
            // const int smem_col = get_swizzled_col(cur_row, smem_col_fragment * ELEMS_PER_VEC4_ACCESS + thread_inner_col);
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(
                    cur_row, c);
 
            // reinterpret_cast<uint32_t *>(
            //     &smem[cur_row * CFG.smem_cols +
            //           smem_col])[0] = regs[r][c];
            reinterpret_cast<uint32_t *>(
                &smem[cur_row * CFG.smem_cols +
                      (smem_col_fragment * ELEMS_PER_VEC4_ACCESS +
                       thread_inner_col)])[0] = regs[r][c];
        }
    }
}

template <typename value_t, int M_fragments, int N_fragments>
__forceinline__ __device__ constexpr void
convert_to_16_bit_dtype(
	float (&src_float)[M_fragments][N_fragments * 2],
    uint32_t (&dest_uint)[M_fragments][N_fragments]) {
    using value2_t =
        std::conditional_t<std::is_same_v<value_t, half>, half2, nv_bfloat162>;
 
    float2(&src)[M_fragments][N_fragments] =
        reinterpret_cast<float2(&)[M_fragments][N_fragments]>(src_float);
    value2_t(&dest)[M_fragments][N_fragments] =
        reinterpret_cast<value2_t(&)[M_fragments][N_fragments]>(dest_uint);
    #pragma unroll
    for (int m = 0; m < M_fragments; ++m) {
        #pragma unroll
        for (int n = 0; n < N_fragments; ++n) {
            if constexpr (std::is_same_v<value_t, half>) {
                dest[m][n] = __float22half2_rn(src[m][n]);
            } else {
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
            }
        }
    }
}
}