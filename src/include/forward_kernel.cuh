#pragma once

#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common.h"
#include "flash_attention.cuh"
#include "gemm.cuh"
#include "ptx_functions.cuh"
#include "softmax.cuh"
#include "static_kernel_configuration.cuh"

#ifdef FA_DEBUG
#include "debug.cuh"
#endif

namespace flash {

#ifndef FA_DEBUG
template <typename Kernel>
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {

    // Prologue: a lot of boilerplate setup
    using accum_t = float;
    using index_t = int64_t;

    using N = typename Kernel::N;

    using value_t = typename Kernel::value_t;

    using Q_t = typename Kernel::Q_t;
    using K_t = typename Kernel::K_t;
    using V_t = typename Kernel::V_t;
 
    // We initialize a CTA for each sample, seq tile, and head.
    const int sample = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq_block = blockIdx.x;
 
    const index_t gmem_seq_stride = args.seq_stride;
 
    const index_t sample_head_offset =
        sample * args.batch_stride + head * args.head_stride;
    // We only read/write one block for Q and O.
    // These offsets are the same for the whole thread-block.
    const index_t QO_gmem_block_offset =
        sample_head_offset + q_seq_block * Kernel::B_r * gmem_seq_stride;
    // We read the entire key sequence.
    const index_t KV_gmem_block_offset = sample_head_offset;
 
    value_t *gmem_Q = &static_cast<value_t *>(args.Q)[QO_gmem_block_offset];
    value_t *gmem_O = &static_cast<value_t *>(args.O)[QO_gmem_block_offset];
    value_t *gmem_K = &static_cast<value_t *>(args.K)[KV_gmem_block_offset];
    value_t *gmem_V = &static_cast<value_t *>(args.V)[KV_gmem_block_offset];
 
    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
    value_t *smem_O = smem_Q;
    value_t *smem_K = smem_Q;
    value_t *smem_V = smem_K;	
 
    // MatrixLDST types
    Q_t Q(gmem_Q, gmem_seq_stride, smem_Q);
    K_t K(gmem_K, gmem_seq_stride, smem_K);
    V_t V(gmem_V, gmem_seq_stride, smem_V);
    // S is only stored in registers.
    typename Kernel::S_accum_t S_accum(nullptr, -1, nullptr);
    // P is only stored in registers.
    typename Kernel::P_value_t P_b16(nullptr, -1, nullptr);
    // The accumulator for O is only kept in registers. At the end of the kernel, it is then converted into a 16-bit type and then copied into gmem.
    typename Kernel::O_accum_t O_accum(nullptr, -1, nullptr);
    typename Kernel::O_value_t O_b16(gmem_O, gmem_seq_stride, smem_O);
 
    // ...
 
    // Start the async copy of the Q and K tiles.
    Q.copy_GM2SM();
    cp_async_commit();
    O_accum.zero();
 
    // Initialize softmax_scale, m, and l.
    const accum_t softmax_scale = rsqrt(static_cast<accum_t>(Kernel::d_head));
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();
    accum_t m[N::QO_fragments_per_warp];
    accum_t l[N::QO_fragments_per_warp];
    #pragma unroll
    for (int q = 0; q < N::QO_fragments_per_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0;
    }
 
    cp_async_wait<0>();
    __syncwarp();
    Q.copy_SM2RF();
    

    // Mainloop: the heart of the algorithm, copy & compute
    for (int j = 0; j < args.n_KV_blocks; ++j) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit();
        S_accum.zero();
        cp_async_wait<0>();
        __syncthreads(); // <---- Barrier 1
        
        K.copy_SM2RF();
 
        matmul<Kernel::S_QK_GEMM>(Q, K, S_accum);
 
        // Online softmax
        accum_t m_next[N::QO_fragments_per_warp];
        scale_S_accum(S_accum.data(), softmax_scale);
        calc_row_max(S_accum.data(), m_next, m);
        scale_l_O(m_next, m, l, O_accum.data());
        exponentiate_tensor(S_accum.data(), m_next);
        update_row_exp_sum(S_accum.data(), l);
 
        // Convert the S accumulator block into P bf16/fp16 input block.
        convert_to_16_bit_dtype<value_t>(S_accum.data(), P_b16.data());
 
        V.copy_GM2SM();
        V.advance_gmem_block();
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads(); // <---- Barrier 2
        V.copy_SM2RF();
 
        matmul<typename Kernel::O_PV_GEMM>(P_b16, V, O_accum);
    }
    
    
    // Epilogue: normalizing the output, converting O from fp32 to the 16-bit, write it back
    final_softmax_normalization(O_accum.data(), l);
 
    convert_to_16_bit_dtype<value_t>(O_accum.data(), O_b16.data());
 
    O_b16.copy_RF2SM();
 
    __syncwarp();
 
    // Copy the final O tile from SMEM to GMEM.
    O_b16.copy_SM2GM();
}

#else

template <typename Kernel>
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {

#ifdef FA_DEBUG
    printf_leader("Kernel start. Grid=(%d,%d,%d), Block=(%d,%d,%d)\n",
                gridDim.x, gridDim.y, gridDim.z,
                blockDim.x, blockDim.y, blockDim.z);
    
    printf_leader("Args: seq_len=%d, n_heads=%d, n_Q_blocks=%d, n_KV_blocks=%d\n",
                args.seq_len, args.n_heads, args.n_Q_blocks, args.n_KV_blocks);
#endif

    // Prologue: a lot of boilerplate setup
    using accum_t = float;
    using index_t = int64_t;

    using N = typename Kernel::N;

    using value_t = typename Kernel::value_t;

    using Q_t = typename Kernel::Q_t;
    using K_t = typename Kernel::K_t;
    using V_t = typename Kernel::V_t;

#ifdef FA_DEBUG
    using O_accum_t = typename Kernel::O_accum_t;
    using O_value_t = typename Kernel::O_value_t;
    using S_accum_t = typename Kernel::S_accum_t;
    using P_value_t = typename Kernel::P_value_t;
#endif
 
    // We initialize a CTA for each sample, seq tile, and head.
    const int sample = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq_block = blockIdx.x;
 
    const index_t gmem_seq_stride = args.seq_stride;
 
    const index_t sample_head_offset =
        sample * args.batch_stride + head * args.head_stride;
    // We only read/write one block for Q and O.
    // These offsets are the same for the whole thread-block.
    const index_t QO_gmem_block_offset =
        sample_head_offset + q_seq_block * Kernel::B_r * gmem_seq_stride;
    // We read the entire key sequence.
    const index_t KV_gmem_block_offset = sample_head_offset;
 
    value_t *gmem_Q = &static_cast<value_t *>(args.Q)[QO_gmem_block_offset];
    value_t *gmem_O = &static_cast<value_t *>(args.O)[QO_gmem_block_offset];
    value_t *gmem_K = &static_cast<value_t *>(args.K)[KV_gmem_block_offset];
    value_t *gmem_V = &static_cast<value_t *>(args.V)[KV_gmem_block_offset];
 
    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
    value_t *smem_O = smem_Q;
    value_t *smem_K = smem_Q;
    value_t *smem_V = smem_K;	
 
    // MatrixLDST types
    Q_t Q(gmem_Q, gmem_seq_stride, smem_Q);
    K_t K(gmem_K, gmem_seq_stride, smem_K);
    V_t V(gmem_V, gmem_seq_stride, smem_V);
    // S is only stored in registers.
    typename Kernel::S_accum_t S_accum(nullptr, -1, nullptr);
    // P is only stored in registers.
    typename Kernel::P_value_t P_b16(nullptr, -1, nullptr);
    // The accumulator for O is only kept in registers. At the end of the kernel, it is then converted into a 16-bit type and then copied into gmem.
    typename Kernel::O_accum_t O_accum(nullptr, -1, nullptr);
    typename Kernel::O_value_t O_b16(gmem_O, gmem_seq_stride, smem_O);
 
#ifdef FA_DEBUG
    // Print kernel launch parameters
    if (is_debug_warp() && is_warp_leader()) {
        printf("=== Flash Attention Kernel Debug ===\n");
        printf("Kernel Config: B_r=%d, B_c=%d, d_head=%d\n", 
               Kernel::B_r, Kernel::B_c, Kernel::d_head);
        printf("Block: (%d, %d, %d), Sample: %d, Head: %d, Q_seq_block: %d\n",
               blockIdx.x, blockIdx.y, blockIdx.z, sample, head, q_seq_block);
        printf("QO_gmem_block_offset: %ld, KV_gmem_block_offset: %ld\n",
               QO_gmem_block_offset, KV_gmem_block_offset);
        printf("gmem_seq_stride: %ld, n_KV_blocks: %d\n", 
               gmem_seq_stride, args.n_KV_blocks);
        
        // 打印N的配置信息
        printf("N::QO_fragments_per_warp: %d\n", N::QO_fragments_per_warp);
        printf("N::d_head_fragments: %d\n", N::d_head_fragments);
    }
#endif
 
    // Start the async copy of the Q and K tiles.
    Q.copy_GM2SM();
    cp_async_commit();
    O_accum.zero();
 
    // Initialize softmax_scale, m, and l.
    const accum_t softmax_scale = rsqrt(static_cast<accum_t>(Kernel::d_head));
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();
    accum_t m[N::QO_fragments_per_warp];
    accum_t l[N::QO_fragments_per_warp];
    #pragma unroll
    for (int q = 0; q < N::QO_fragments_per_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0;
    }
 
    cp_async_wait<0>();
    __syncwarp();
    Q.copy_SM2RF();
    
#ifdef FA_DEBUG
    // Print Q after loading
    if (is_debug_warp()) {
        // constexpr int Q_smem_rows = Q_LDST.warp_ldst_rows;
        // constexpr int Q_smem_cols = Q_LDST.smem_cols;
        // print_smem_matrix<value_t>(
        //     Q.smem_gsm_ptr,
        //     Q_smem_cols,
        //     min(32, Q_smem_rows),
        //     min(32, Q_smem_cols),
        //     "Q_GMEM2SMEM",
        //     0
        // );
    
        // 打印Q从SMEM到RF后的数据
        constexpr int q_rows = Q_t::matrix_storage_t::rows;
        constexpr int q_cols = Q_t::matrix_storage_t::cols;
        print_rf_matrix<q_rows, q_cols>(
            reinterpret_cast<uint32_t(&)[q_rows][q_cols]>(Q.data()),
            "Q_RF",
            0
        );
    }
#endif

    // Mainloop: the heart of the algorithm, copy & compute
    for (int j = 0; j < args.n_KV_blocks; ++j) {
#ifdef FA_DEBUG
        // Print iteration info
        printf_leader("--- KV Block Iteration j=%d (total=%d) ---\n", j, args.n_KV_blocks);
#endif
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit();
        S_accum.zero();
        cp_async_wait<0>();
        __syncthreads(); // <---- Barrier 1
        
        K.copy_SM2RF();

#ifdef FA_DEBUG
        // 打印K从GMEM到SMEM后的数据
        if (is_debug_warp()) {
            constexpr int k_rows = K_t::matrix_storage_t::rows;
            constexpr int k_cols = K_t::matrix_storage_t::cols;
            print_rf_matrix<k_rows, k_cols>(
                reinterpret_cast<uint32_t(&)[k_rows][k_cols]>(K.data()),
                "K_RF",
                j
            );
        }
#endif
 
        matmul<Kernel::S_QK_GEMM>(Q, K, S_accum);

#ifdef FA_DEBUG
        // Print S_accum after matmul (before softmax)
        if (is_debug_warp()) {
            // S_accum是accum_t类型（float），使用不同的打印函数
            constexpr int S_rows = S_accum_t::matrix_storage_t::rows;
            constexpr int S_cols = S_accum_t::matrix_storage_t::cols;
            
            // 注意：S_accum.data()返回的是accum_t(&)[rows][cols]
            // 但print_rf_accum_matrix期望的是float数组
            auto& s_data = S_accum.data();
            print_rf_accum_matrix<S_rows, S_cols>(
                reinterpret_cast<float(&)[S_rows][S_cols]>(s_data),
                "S_accum",
                j
            );
        }
#endif
 
        // Online softmax
        accum_t m_next[N::QO_fragments_per_warp];
        scale_S_accum(S_accum.data(), softmax_scale);
        calc_row_max(S_accum.data(), m_next, m);
        scale_l_O(m_next, m, l, O_accum.data());
        exponentiate_tensor(S_accum.data(), m_next);
        update_row_exp_sum(S_accum.data(), l);
 
        // Convert the S accumulator block into P bf16/fp16 input block.
        convert_to_16_bit_dtype<value_t>(S_accum.data(), P_b16.data());

#ifdef FA_DEBUG
        // Print P_b16 after conversion
        if (is_debug_warp()) {
            printf("After softmax - m_next: ");
            print_rf_row<N::QO_fragments_per_warp, accum_t>(
                m_next,
                "m_next",
                j,
                false
            );
            print_rf_row<N::QO_fragments_per_warp, accum_t>(
                l,
                "l_after_softmax",
                j,
                false
            );
        }
        
        // 打印P_b16（softmax后的概率矩阵）
        if (is_debug_warp()) {
            constexpr int p_rows = P_value_t::matrix_storage_t::rows;
            constexpr int p_cols = P_value_t::matrix_storage_t::cols;
            print_rf_matrix<p_rows, p_cols>(
                reinterpret_cast<uint32_t(&)[p_rows][p_cols]>(P_b16.data()),
                "P_b16",
                j
            );
        }
#endif

        V.copy_GM2SM();
        V.advance_gmem_block();
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads(); // <---- Barrier 2

// #ifdef FA_DEBUG
//         // Print V after loading
//         if (is_debug_warp()) {
//             constexpr int V_smem_rows = V_LDST.warp_ldst_rows;
//             constexpr int V_smem_cols = V_LDST.smem_cols;
//             print_smem_matrix<value_t>(
//                 V.smem_gsm_ptr,
//                 V_smem_cols,
//                 min(32, V_smem_rows),
//                 min(32, V_smem_cols),
//                 "V_GMEM2SMEM",
//                 j
//             );
//         }
// #endif

        V.copy_SM2RF();

#ifdef FA_DEBUG
        // 打印V从SMEM到RF后的数据
        if (is_debug_warp()) {
            constexpr int v_rows = V_t::matrix_storage_t::rows;
            constexpr int v_cols = V_t::matrix_storage_t::cols;

            print_rf_matrix<v_rows, v_cols>(
                reinterpret_cast<uint32_t(&)[v_rows][v_cols]>(V.data()),
                "V_RF",
                j
            );
        }
#endif

        matmul<typename Kernel::O_PV_GEMM>(P_b16, V, O_accum);

#ifdef FA_DEBUG
        // Print O_accum after matmul
        if (is_debug_warp()) {
            constexpr int O_rows = O_accum_t::matrix_storage_t::rows;
            constexpr int O_cols = O_accum_t::matrix_storage_t::cols;

            auto& o_data = O_accum.data();
            flash::print_rf_accum_matrix<O_rows, O_cols>(
                reinterpret_cast<float(&)[O_rows][O_cols]>(o_data),
                "O_accum",
                j
            );
        }
#endif
    }
    
#ifdef FA_DEBUG
    // Print final O_accum before normalization
    printf_leader("--- Final O_accum before normalization ---\n");

    if (is_debug_warp()) {
        print_rf_row<N::QO_fragments_per_warp, accum_t>(
            l,
            "l_final",
            args.n_KV_blocks,
            false
        );

        // print_rf_accum_matrix<O_rf_rows, O_rf_cols>(
        //     O_accum.data(),
        //     "O_accum_final",
        //     args.n_KV_blocks
        // );
    }
#endif

    // Epilogue: normalizing the output, converting O from fp32 to the 16-bit, write it back
    final_softmax_normalization(O_accum.data(), l);

#ifdef FA_DEBUG
    printf_leader("--- Final O_accum after normalization ---\n");

    // Print O_accum after normalization
    if (is_debug_warp()) {
        print_rf_row<N::QO_fragments_per_warp, accum_t>(
            l,
            "l_final_norm",
            args.n_KV_blocks,
            false
        );
        // print_rf_accum_matrix<O_rf_rows, O_rf_cols>(
        //     O_accum.data(),
        //     "O_accum_final_norm",
        //     args.n_KV_blocks
        // );
    }
#endif
 
    convert_to_16_bit_dtype<value_t>(O_accum.data(), O_b16.data());

#ifdef FA_DEBUG
    // Print O_b16 before writing to SMEM
    if (is_debug_warp()) {
        constexpr int ob_rows = O_value_t::matrix_storage_t::rows;
        constexpr int ob_cols = O_value_t::matrix_storage_t::cols;

        print_rf_matrix<ob_rows, ob_cols>(
            reinterpret_cast<uint32_t(&)[ob_rows][ob_cols]>(O_b16.data()),
            "O_b16_final",
            args.n_KV_blocks
        );
    }
#endif
 
    O_b16.copy_RF2SM();
 
    __syncwarp();

#ifdef FA_DEBUG
    // Print SMEM O before writing to GMEM
    // if (is_debug_warp()) {
    //     constexpr int O_smem_rows = O_LDST.warp_ldst_rows;
    //     constexpr int O_smem_cols = O_LDST.smem_cols;
    //     print_smem_matrix<value_t>(
    //         O_b16.smem_gsm_ptr,
    //         O_smem_cols,
    //         min(32, O_smem_rows),
    //         min(32, O_smem_cols),
    //         "O_RF2SMEM",
    //         args.n_KV_blocks
    //     );
    // }
#endif
 
    // Copy the final O tile from SMEM to GMEM.
    O_b16.copy_SM2GM();

#ifdef FA_DEBUG
    // Print completion message
    if (is_debug_warp() && is_warp_leader()) {
        printf("=== Flash Attention Kernel Complete ===\n");
        printf("Processed %d KV blocks for block (%d, %d, %d)\n", 
               args.n_KV_blocks, blockIdx.x, blockIdx.y, blockIdx.z);
        printf("Output written to GMEM at offset %ld\n", QO_gmem_block_offset);
    }
#endif
}
#endif
}
