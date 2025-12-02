# flash_attention_practice

## Introduction

A [Flash Attention 2](https://arxiv.org/abs/2307.08691) implementation following the instruction of [Flash Attention From Scratch](https://lubits.ch/flash/).

## Testing

### Installation

Normal mode

```bash
pip install --no-build-isolation . # `flash_attention` (CUDA kernels)
pip install ./py # `flash_helpers` (utils and kernel configuration) python packages
```

Debug mode

```bash
export FA_DEBUG='true'

pip install --no-build-isolation . # `flash_attention` (CUDA kernels)
pip install ./py # `flash_helpers` (utils and kernel configuration) python packages
```

### Debug

Compile via debug mode

```bash
export KERNELS="64,64"
python tools/debug/sanity_check.py
```

### Benchmarking

```bash
export KERNELS="64,64"
seq_lens="${1:-1024,2048,4096}"

python tools/benchmark/pt_bench.py \
                --d_heads 128 \
                --seq_lens="${seq_lens}"\
                --num_warmups 1 \
                --num_repeats 2

python tools/benchmark/pt_bench.py \
                --d_heads 128 \
                --seq_lens=2048\
                --num_warmups 1 \
                --num_repeats 2
```

## Results

Performances of each kernel iteration.
