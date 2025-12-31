#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common.h"
#include "ptx_functions.cuh"

namespace flash_practice {

// Swizzle

template <int col_fragments>
__forceinline__ __device__ constexpr int swizzled_col_fragment(int row, int col_fragment) {
    static_assert(col_fragments % ELEMS_PER_VEC4_ACCESS == 0,
                  "# col tiles is a multiple of # elems");

    // The % ELEMS_PER_VEC4_ACCESS makes sure that the swizzled column stays
    // within the same 8 element window.
    return (row % ELEMS_PER_VEC4_ACCESS) ^ col_fragment;
}

template <int col_fragments>
__forceinline__ __device__ constexpr int get_smem_col_fragment(const int row,
                                              const int col_fragment) {
    return swizzled_col_fragment<col_fragments>(row, col_fragment);
}


// GMEM <-> SMEM

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
          int ROW_FRAGMENTS,
          int SMEM_COL,
          typename value_t,
          typename index_t = int64_t>
__forceinline__ __device__ constexpr void copy_block_GSM(
	value_t *gmem,
	value_t *smem,
    index_t gmem_seq_stride,
    const int lane_id) {

    constexpr int n_row_iters =
        ROW_FRAGMENTS * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;
 
    constexpr int col_fragments_per_iter = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    constexpr int col_fragments_per_row = SMEM_COL / COLS_PER_FRAGMENT;
 
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
                get_smem_col_fragment<col_fragments_per_row>(
                                    cur_row, gmem_col_fragment);
 
            op()(&gmem[cur_row * gmem_seq_stride +
                       gmem_col_fragment * COLS_PER_FRAGMENT],
                 &smem[SMEM_COL +
                       smem_col_fragment * COLS_PER_FRAGMENT]);
        }
    }
}


// SMEM <-> RF

template <typename value_t, int n_copies, int row_fragments, int col_fragments>
struct RFMatrix {
    using storage_t = std::conditional_t<sizeof(value_t) == 4, float, uint32_t>;
    static constexpr int regs_per_fragment = sizeof(value_t) / 2;
    static constexpr int rows = row_fragments;
    static constexpr int cols = col_fragments * regs_per_fragment;
 
    storage_t regs[n_copies][rows][cols];
 
    __forceinline__ __device__ constexpr storage_t (&data(const int stage = 0))[rows][cols] {
        return reinterpret_cast<storage_t(&)[rows][cols]>(regs[stage]);
    }
 
    __forceinline__ __device__ constexpr void zero() {
        FA_UNROLL
        for (int i = 0; i < n_copies; ++i) {
            FA_UNROLL
            for (int j = 0; j < rows; ++j) {
                FA_UNROLL
                for (int k = 0; k < cols; ++k) {
                    regs[i][j][k] = 0;
                }
            }
        }
    }
};

// Q and Kj
template <int ROW_FRAGMENTS, int COL_FRAGMENTS, int SMEM_COL, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_SM2RF(
    uint32_t (&regs)[ROW_FRAGMENTS][COL_FRAGMENTS],
    value_t *smem,
    const int lane_id,
    const int col_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;
 
    constexpr int col_fragments = SMEM_COL / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;
 
    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;
 
    #pragma unroll
    for (int r = 0; r < ROW_FRAGMENTS; r += row_fragments_per_iter) {
        const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
        #pragma unroll
        for (int c = 0; c < COL_FRAGMENTS; c += col_fragments_per_iter) {
            // Use swizzled addresses to match the layout from GMEM→SMEM transfers
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments>(
                                    cur_row, thread_col_fragment + c + col_fragment_offset);
 
            ldmatrix_x4(&smem[cur_row * SMEM_COL +
                        smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                        regs[r][c], regs[r + 1][c], regs[r][c + 1],
                        regs[r + 1][c + 1]);
        }
    }
}

// template <int ROW_FRAGMENTS, int COL_FRAGMENTS, int SMEM_COL, typename value_t>
// __forceinline__ __device__ constexpr void copy_warp_fragment_SM2RF_optimized(
//     uint32_t (&regs)[ROW_FRAGMENTS][COL_FRAGMENTS],
//     value_t *smem,
//     const int lane_id,
//     const int col_fragment_offset = 0) {
//     constexpr int row_fragments_per_iter = 2;
//     constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;
 
//     constexpr int col_fragments = SMEM_COL / ELEMS_PER_VEC4_ACCESS;
//     constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;
    
//     // 确保 col_fragments_per_iter 是 2 的幂，这样我们可以安全使用 XOR
//     static_assert((col_fragments_per_iter & (col_fragments_per_iter - 1)) == 0, 
//                   "col_fragments_per_iter must be power of 2 for optimization");
 
//     const int thread_row = lane_id % rows_per_iter;
//     const int thread_col_fragment = lane_id / rows_per_iter;
 
//     #pragma unroll
//     for (int r = 0; r < ROW_FRAGMENTS; r += row_fragments_per_iter) {
//         const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
        
//         // 预计算 row_mod 和基础 XOR 值
//         const int row_mod = cur_row % ELEMS_PER_VEC4_ACCESS;
//         const int base_col = thread_col_fragment + col_fragment_offset;
        
//         // 预计算第一个片段的 XOR 值
//         int current_xor = row_mod ^ base_col;
        
//         #pragma unroll
//         for (int c = 0; c < COL_FRAGMENTS; c += col_fragments_per_iter) {
//             // 使用预计算的 XOR 值
//             const int smem_col_fragment = current_xor;
            
//             ldmatrix_x4(&smem[cur_row * SMEM_COL +
//                         smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
//                         regs[r][c], regs[r + 1][c], regs[r][c + 1],
//                         regs[r + 1][c + 1]);
            
//             // 为下一个片段更新 XOR 值
//             // 注意：这里使用 XOR 而不是加法，因为 col_fragments_per_iter 是 2 的幂
//             // 且我们假设没有跨 8 的边界（因为 row_mod 和 base_col 都在 0-7 范围内）
//             current_xor ^= col_fragments_per_iter;
//         }
//     }
// }

// Vj
template <int ROW_FRAGMENTS, int COL_FRAGMENTS, int SMEM_COL, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[ROW_FRAGMENTS][COL_FRAGMENTS],
    value_t *smem,
    const int lane_id,
    const int row_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;
 
    constexpr int col_fragments = SMEM_COL / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;
 
    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;
 
    #pragma unroll
    for (int r = 0; r < COL_FRAGMENTS; r += row_fragments_per_iter) {
        const int cur_row =
            thread_row + (r + row_fragment_offset) * ROWS_PER_FRAGMENT;
        #pragma unroll
        for (int c = 0; c < ROW_FRAGMENTS; c += col_fragments_per_iter) {
            const int smem_col_fragment = 
                get_smem_col_fragment<col_fragments>(
                    cur_row, thread_col_fragment + c);
 
            ldmatrix_x4_transpose(
                &smem[cur_row * SMEM_COL +
                      smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                regs[c][r], regs[c][r + 1], regs[c + 1][r], regs[c + 1][r + 1]);
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

} // flash_practice
