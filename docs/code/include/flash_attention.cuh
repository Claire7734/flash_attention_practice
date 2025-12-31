#include "common.h"

#include <cuda_bf16.h>
#include <cstdint>
#include <float.h>
#include <iostream>

#include "common.h"
#include "ldst.cuh"
#include "ptx_functions.cuh"
#include "calculate.cuh"


namespace flash_practice {

constexpr int smem_bytes(int B_r, int B_c, int d_head, int elem_size = 2) {
    return (B_r + B_c * 2) * d_head * elem_size;
}

struct ForwardKernelArgs {
    using index_t = int64_t;

    void *__restrict__ Q;
    void *__restrict__ K;
    void *__restrict__ V;
    void *__restrict__ O;

    // We assume all strides are the same across all inputs, and that
    // the tensors are all row major.
    const index_t batch_stride;
    const index_t seq_stride;
    const index_t head_stride;

    const index_t seq_len;
    const index_t n_heads;

    const int n_Q_blocks;
    const int n_KV_blocks;
};


struct KernelConfig {
    const int d_head = 128;  // [128]
    const int B_r = 64;     // [64, 128]
    const int B_c = 64;     // [32, 64, 128]
    const int n_warps = 4; // [4, 8]

    c10::ScalarType dtype = torch::kFloat16;

    int smem_bytes(int elem_size = 2) const {
        return (B_r + B_c * 2) * d_head * elem_size;
    }

    int num_ctas_per_sm(int max_smem_bytes) const {
        // The max # ctas will be 2 or less due to register limits.
        if ((n_warps == 8) || (max_smem_bytes < smem_bytes() * 2)) {
            return 1;
        }

        return 2;
    }
};

template<int B_r, int B_c, int D_HEAD, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {

    using accum_t = float;
    using index_t = int64_t;
    using value_t = nv_bfloat16;

    constexpr int async = true;
    constexpr bool optimized_softmax = true;

    const int sample = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq_block = blockIdx.x;

    const int tid = threadIdx.x;
    const int warp_rank = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const index_t gmem_seq_stride = args.seq_stride;

    const int num_kv_iter = args.n_KV_blocks;
 
    // const auto batch_stride = TQ.stride(0);
    // const auto seq_stride = TQ.stride(1);
    // const auto head_stride = TQ.stride(2);

    const index_t sample_head_offset =
        sample * args.batch_stride + head * args.head_stride;
    // We only read/write one block for Q and O.
    // These offsets are the same for the whole thread-block.
    const index_t QO_gmem_block_offset =
        sample_head_offset + q_seq_block * B_r * gmem_seq_stride;
    // We read the entire key sequence.
    const index_t KV_gmem_block_offset = sample_head_offset; // each segment advances BLOCK_KV * gmem_seq_stride

    value_t *gmem_Q = &static_cast<nv_bfloat16*>(args.Q)[QO_gmem_block_offset];
    value_t *gmem_K = &static_cast<nv_bfloat16*>(args.K)[KV_gmem_block_offset];
    value_t *gmem_V = &static_cast<nv_bfloat16*>(args.V)[KV_gmem_block_offset];
    value_t *gmem_O = &static_cast<nv_bfloat16*>(args.O)[QO_gmem_block_offset];

    // shared memory
    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
    value_t *smem_O = smem_Q; // Q & O share the same smem space
    value_t *smem_K = &smem_Q[B_r * D_HEAD];
    value_t *smem_V = &smem_K[B_c * D_HEAD];


    // The number of d_head tiles loaded and operated on by this thread
    // block.
    constexpr int d_head_fragments = D_HEAD / COLS_PER_FRAGMENT;

    // each warp holds a sub-tile (B_r / NUM_WARPS, D_HEAD) of Q 
    // while sharing entire tile (B_c, D_HEAD) of K & V
    constexpr int QO_rows_per_warp = B_r / NUM_WARPS;
    constexpr int QO_fragments_per_warp = QO_rows_per_warp / ROWS_PER_FRAGMENT;

    // For a K/V block, each warp will independently load a chunk of the (B_c,
    // d_head), but perform computations on the entire block loaded by the
    // thread-block.
    constexpr int KV_calc_fragments = B_c / ROWS_PER_FRAGMENT;
    constexpr int KV_ldst_fragments_per_warp = KV_calc_fragments / NUM_WARPS;
    constexpr int KV_ldst_rows_per_warp = KV_ldst_fragments_per_warp * ROWS_PER_FRAGMENT;

    // RF storage
    // n_copies = 1 as not using double buffering for SMEM -> RF to avoid register spilling
    constexpr int n_copies = 1; 
    RFMatrix<value_t, n_copies, QO_fragments_per_warp, MMA_LOAD_FRAGMENTS> rfmatrix_Q;
    RFMatrix<value_t, n_copies, KV_calc_fragments, MMA_LOAD_FRAGMENTS> rfmatrix_K;
    RFMatrix<value_t, n_copies, d_head_fragments, MMA_LOAD_FRAGMENTS> rfmatrix_V_T;

    RFMatrix<accum_t, 1, QO_fragments_per_warp, KV_calc_fragments> rfmatrix_S_accum;
    RFMatrix<value_t, 1, QO_fragments_per_warp, KV_calc_fragments> rfmatrix_P_b16;
    RFMatrix<accum_t, 1, QO_fragments_per_warp, d_head_fragments> rfmatrix_O_accum;
    RFMatrix<value_t, 1, QO_fragments_per_warp, d_head_fragments> rfmatrix_O_b16;

    const int QO_warp_seq = QO_rows_per_warp * warp_rank;
    copy_block_GSM<GM2SM_async<value_t>, QO_fragments_per_warp, D_HEAD, value_t>(
        gmem_Q, 
        smem_Q + QO_warp_seq * D_HEAD, 
        gmem_seq_stride, 
        lane_id);
    cp_async_commit();

    const int KV_warp_seq = KV_ldst_rows_per_warp * warp_rank;
    auto load_K = [&](int kv_id) {
        // KV_ldst_fragments_per_warp = 2, D_HEAD = 128
        copy_block_GSM<GM2SM_async<value_t>, KV_ldst_fragments_per_warp, D_HEAD, value_t>(
            gmem_K, 
            smem_K + KV_warp_seq * D_HEAD, 
            gmem_seq_stride, 
            lane_id);
        gmem_K += B_c * gmem_seq_stride;
        cp_async_commit();
    };

    auto load_V = [&](int kv_id) {
        copy_block_GSM<GM2SM_async<value_t>, KV_ldst_fragments_per_warp, D_HEAD, value_t>(
            gmem_V, 
            smem_V + KV_warp_seq * D_HEAD, 
            gmem_seq_stride, 
            lane_id);
        gmem_V += B_c * gmem_seq_stride;
        cp_async_commit();
    };

    // prefetch K
    load_K(0);

    rfmatrix_O_accum.zero();

    // Initialize softmax_scale, m, and l.
    const accum_t softmax_scale = rsqrt(static_cast<accum_t>(D_HEAD)) *
                                  (optimized_softmax ? M_LOG2E : 1.0);
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();
    accum_t m[QO_fragments_per_warp];
    accum_t l[QO_fragments_per_warp];
    #pragma unroll
    for (int q = 0; q < QO_fragments_per_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0;
    }

    // Only wait for Q, and leave K in flight
    cp_async_wait<1>();
    __syncwarp();

    for (int kv_id = 0; kv_id < num_kv_iter; ++kv_id) {

        rfmatrix_S_accum.zero();

        // Wait untile tile K finishes transfering
        cp_async_wait<0>();
        // Need the entire (64, 128) K, which is handled 
        // at CTA level with all warps
        __syncthreads();

        load_V(kv_id);

        // MMA S = Q @ K.T
        // loop - d_head / 8 for QK^T
        #pragma unroll
        for (int k = 0; k < KV_calc_fragments; k += MMA_LOAD_FRAGMENTS) {
            // Load a sub-tile of Q along k-dimension as MMA_LOAD_FRAGMENTS
            // QO_fragments_per_warp = 2
            copy_warp_fragment_SM2RF<QO_fragments_per_warp, MMA_LOAD_FRAGMENTS, D_HEAD, value_t>(
                        rfmatrix_Q.data(), smem_Q + QO_warp_seq * D_HEAD, lane_id, k);

            // KV_calc_fragments = 8
            copy_warp_fragment_SM2RF<KV_calc_fragments, MMA_LOAD_FRAGMENTS, D_HEAD, value_t>(
                        rfmatrix_K.data(), smem_K, lane_id, k);
            
            warp_fragment_mma_f32_accum<value_t>(rfmatrix_Q.data(), rfmatrix_K.data(), rfmatrix_S_accum.data(),
                                        0, 0);
        }

        // Wait untile tile V finishes transfering
        cp_async_wait<0>();
        __syncthreads();

        load_K(kv_id + 1);

        // Online softmax
        
        accum_t m_next[QO_fragments_per_warp];
        if constexpr (!optimized_softmax) {
            scale_S_accum(rfmatrix_S_accum.data(), softmax_scale);
        }
        calc_row_max(rfmatrix_S_accum.data(), m_next, m);
        scale_l_O<optimized_softmax>(m_next, m, l, rfmatrix_O_accum.data(),
                                             softmax_scale);
        exponentiate_tensor<optimized_softmax>(rfmatrix_S_accum.data(), m_next,
                                                       softmax_scale);
        update_row_exp_sum(rfmatrix_S_accum.data(), l);
 
        // Convert the S accumulator block into P bf16/fp16 input block.
        convert_to_16_bit_dtype<value_t>(rfmatrix_S_accum.data(), rfmatrix_P_b16.data());

        // MMA O = P @ V
        // loop - B_c / 8    for PV
        #pragma unroll
        for (int k = 0; k < d_head_fragments; k += MMA_LOAD_FRAGMENTS) {
            // No need to load P as it's in RF

            // KV_calc_fragments = 8
            copy_warp_fragment_transposed_SM2RF<d_head_fragments, MMA_LOAD_FRAGMENTS, D_HEAD, value_t>(
                        rfmatrix_V_T.data(), smem_V, lane_id, k);
            
            warp_fragment_mma_f32_accum<value_t>(rfmatrix_P_b16.data(), rfmatrix_V_T.data(), rfmatrix_O_accum.data(),
                                        k, 0);
        }
    }

    // O_b16.copy_RF2SM();
    // __syncwarp();
    // O_b16.copy_SM2GM();
}

} // flash_practice
