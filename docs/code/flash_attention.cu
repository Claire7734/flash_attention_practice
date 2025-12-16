#include <torch/python.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <utility>
#include <vector>
#include "cuda_utils.cuh"
#include "flash_attention.cuh"

using namespace flash_practice;

std::tuple<torch::Tensor, float> 
flash_attention_forward(
    const torch::Tensor &TQ,
    const torch::Tensor &TK, 
    const torch::Tensor &TV,
    std::optional<at::Tensor> &out_,
    bool benchmark = false) {

    CHECK_INPUT(TQ);
    CHECK_INPUT(TK);
    CHECK_INPUT(TV);

    at::cuda::CUDAGuard device_guard{TQ.device()};
    const int compute_capability =
        cuda_device_compute_capability(TQ.device().index());
    TORCH_CHECK(compute_capability >= 80,
                "Flash Attention requires SM_80 or higher (current: SM_",
                compute_capability / 10, ".", compute_capability % 10, ")");

    // Check data types
    const auto Q_dtype = TQ.dtype();
    TORCH_CHECK(Q_dtype == torch::kFloat16 || Q_dtype == torch::kBFloat16,
                "Only fp16 and bf16 are supported");
    TORCH_CHECK(TK.dtype() == Q_dtype,
                "Input tensors must have the same data type");
    TORCH_CHECK(TV.dtype() == Q_dtype,
                "Input tensors must have the same data type");

    const int d_head = 128;  // [128]
    const int B_r = 64;     // [64, 128]
    const int B_c = 64;     // [32, 64, 128]
    const int n_warps = 4; // [4, 8]

    const auto batch_size = TQ.size(0);
    const auto seq_len = TQ.size(1);
    const auto n_heads = TQ.size(2);
    const auto input_d_head = TQ.size(3);
    TORCH_CHECK(input_d_head == d_head, 
                "Input head dimension must be ", d_head, " (got ", input_d_head, ")");

    // Only supported configuration currently.
    TORCH_CHECK(TQ.sizes() == TK.sizes(),
                "Query and key tensors have same shape");
    TORCH_CHECK(TQ.sizes() == TV.sizes(),
                "Query and value tensors have same shape");

    TORCH_CHECK(seq_len % B_r == 0,
                "Only multiples of B_r are supported for seq_len Q currently");
    TORCH_CHECK(seq_len % B_c == 0,
                "Only multiples of B_c are supported for seq_len K currently");

    const auto batch_stride = TQ.stride(0);
    const auto seq_stride = TQ.stride(1);
    const auto head_stride = TQ.stride(2);

    torch::Tensor TO;
    if (out_.has_value()) {
        TO = out_.value();
        TORCH_CHECK(TO.dtype() == Q_dtype,
                    "Output tensor must have the same dtype as inputs");

        TORCH_CHECK(TQ.sizes() == TV.sizes(),
                    "Query and output tensors have same shape");
    } else {
        TO = torch::empty_like(TQ);
    }

    const int n_Q_blocks = CEIL_DIV(seq_len, B_r);
    const int n_KV_blocks = CEIL_DIV(seq_len, B_c);
    const int n_threads = n_warps * WARP_SIZE;

    ForwardKernelArgs args{TQ.data_ptr(), TK.data_ptr(), TV.data_ptr(),
                           TO.data_ptr(), batch_stride,  seq_stride,
                           head_stride,   seq_len,       n_heads,
                           n_Q_blocks,    n_KV_blocks};

    dim3 blockDim(n_threads);
    dim3 gridDim{static_cast<uint>(n_Q_blocks), static_cast<uint>(n_heads),
                 static_cast<uint>(batch_size)};

    float runtime = 0.0f;
    cudaEvent_t start, stop;

    const int elem_size = (Q_dtype == torch::kFloat16) ? 2 : 2;
    const int smem_bytes_val = smem_bytes(B_r, B_c, d_head, elem_size);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (smem_bytes_val > 48 * 1024) {
        cudaFuncSetAttribute(
            flash_forward_kernel<B_r, B_c, d_head, n_warps>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes_val);
    }

    if (benchmark) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);
    }

    flash_forward_kernel<B_r, B_c, d_head, n_warps><<<gridDim, blockDim, smem_bytes_val, stream>>>(args);

    if (benchmark) {
        cudaEventRecord(stop, stream);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runtime, start, stop);
    }

    return std::make_tuple(TO, runtime);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("o"),
          py::arg("benchmark") = false, "Flash Attention forward (CUDA)");
}