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

namespace flash {

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
}
