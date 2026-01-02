import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
from triton.testing import do_bench

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

CURRENT_DIR = Path(__file__).parent

include_dir = str(CURRENT_DIR / "include")

def get_nvcc_compile_args(debug=False):
    """获取NVCC编译参数"""
    include_dir = str(CURRENT_DIR / "include")
    
    base_flags = [
        f"-I{include_dir}",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "--use_fast_math",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--generate-line-info",
        "--ptxas-options=-v",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
        "-Xcompiler", "-fPIC",
    ]
    
    if debug:
        base_flags.extend(["-g", "-G", "-O0"])  # 调试模式
    else:
        base_flags.append("-O3")  # 发布模式
    
    return base_flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--nh", type=int, default=32)
    parser.add_argument("--lq", type=int, default=4096)
    parser.add_argument("--lkv", type=int, default=4096)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    flash_attention_cuda = load(
        name="flash_attention_cuda",
        sources=[str(CURRENT_DIR / "flash_attention.cu")],
        extra_cuda_cflags=get_nvcc_compile_args(args.debug),
        verbose=True,
    )

    bs = args.bs
    nh = args.nh
    lq = args.lq
    lkv = args.lkv
    head_dim = 128

    if lq % 64 != 0 or lkv % 64 != 0:
        print(f"警告: 序列长度必须能被64整除，lq={lq}, lkv={lkv}")
        # 调整到最近的64的倍数
        lq = ((lq + 63) // 64) * 64
        lkv = ((lkv + 63) // 64) * 64
        print(f"已调整为: lq={lq}, lkv={lkv}")

    # add a small offset so that output does not have a mean of zero,
    # which will result in large relative error
    def generate_input(*shape):
        return torch.randn(shape).add(0.5).to(torch.float16).cuda()

    Q = generate_input(bs, lq, nh, head_dim)
    K = generate_input(bs, lkv, nh, head_dim)
    V = generate_input(bs, lkv, nh, head_dim) 
    
    Q_sdpa = Q.permute(0, 2, 1, 3).contiguous()
    K_sdpa = K.permute(0, 2, 1, 3).contiguous()
    V_sdpa = V.permute(0, 2, 1, 3).contiguous()

    if args.profile is not None:
        if args.profile == "fa":
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                F.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa)

        elif args.profile == "cudnn":
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                F.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa)

        else:
            flash_attention_cuda.forward(Q, K, V, None, False)

        torch.cuda.synchronize()
        return

    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 209.5,
        "NVIDIA GeForce RTX 3090": 67.29,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), 0)

    results = []

    def bench_and_print(f, name, *args):
        # sleep to stabilize thermal
        time.sleep(1)

        latency_ms = do_bench(lambda: f(*args), return_mode="median")
        tflops = 4 * bs * nh * lq * lkv * head_dim / latency_ms / 1e9
        pct_sol = tflops / sol * 100
        results.append([name, round(latency_ms, 4), round(tflops, 2), round(pct_sol, 2)])

    out_ref_sdpa = F.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa)
    out_ref = out_ref_sdpa.permute(0, 2, 1, 3) # (bs, lq, nh, head_dim)

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        bench_and_print(F.scaled_dot_product_attention, "F.sdpa() - FA", 
                        Q_sdpa, K_sdpa, V_sdpa)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        bench_and_print(F.scaled_dot_product_attention, "F.sdpa() - CuDNN", 
                        Q_sdpa, K_sdpa, V_sdpa)

    if flash_attn_func is not None:
        out_flash = flash_attn_func(Q, K, V)
        out_flash_sdpa = out_flash.permute(0, 2, 1, 3)
        torch.testing.assert_close(out_flash_sdpa, out_ref_sdpa, rtol=1e-3, atol=1e-3)
        bench_and_print(
            flash_attn_func,
            "flash-attn",
            Q, K, V
        )

    def call_flash_attention(q, k, v, benchmark=True):
        output, _ = flash_attention_cuda.forward(q, k, v, None, benchmark)
        return output
    
    out_custom = call_flash_attention(Q, K, V, benchmark=False)
    try:
        torch.testing.assert_close(out_custom, out_ref, rtol=1e-2, atol=1e-3)
        print("✓ 自定义实现输出与参考输出匹配")
    except AssertionError as e:
        print(f"⚠ 自定义实现输出与参考输出有差异: {e}")
        # 计算相对误差
        rel_error = torch.mean(torch.abs(out_custom - out_ref) / torch.abs(out_ref)).item()
        print(f"   平均相对误差: {rel_error:.6f}")

    bench_and_print(call_flash_attention, "Custom Flash Attention", Q, K, V)

    df = pd.DataFrame(results, columns=["Kernel", "Latency (ms)", "TFLOPS", "% SOL"])
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
