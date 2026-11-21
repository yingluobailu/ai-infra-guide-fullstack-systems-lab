# Day 13 — CUDA Vectorized Reduction

## Objective

Optimize GPU sum reduction using vectorized memory
access and Grid-stride Loop.

## Implementations

- V6: Warp Shuffle reduction baseline
- V7: float4 + Grid-stride Loop + Warp Shuffle
- Final: Two-stage reduction of partial sums

## Environment

- GPU: NVIDIA GeForce RTX 3090
- CUDA architecture: sm_86
- SM count: 82
- Block size: 256
- V7 target grid: 328 blocks

## Build and Run

```bash
mkdir -p build/day13

nvcc -std=c++17 -O3 -arch=sm_86 \
    lessons/day13/vectorized_reduce.cu \
    -o build/day13/vectorized_reduce

./build/day13/vectorized_reduce
```

## Correctness

14 input sizes tested, including small arrays,
non-multiples of 4, and large inputs.

Results:
- V6: ALL PASS
- V7: ALL PASS
- Compute Sanitizer: 0 errors

## Performance

Input: 134,217,728 float32 elements (512 MiB)
Timing: CUDA Events, including final reduction
Warmup: 10
Iterations: 50

| Version | Time (ms) | Effective BW (GB/s) |
|---------|-----------|---------------------|
| V6 | 0.93208 | 575.99 |
| V7 | 0.60783 | 883.26 |

V7 speedup over V6: 1.533x

Effective bandwidth is calculated using input
bytes only, not measured total DRAM traffic.

## Key Findings

1. float4 enables vectorized loading of four
   consecutive float32 elements per load.
2. Vectorized access requires 16-byte alignment.
3. Tail handling is needed for non-multiples of 4.
4. Grid-stride Loop separates the amount of data
   from the number of launched blocks.
5. Warp Shuffle supports efficient intra-warp
   reduction, while shared memory connects warps.
6. Performance gains depend on input size,
   grid configuration, and hardware.
7. V7 improves performance by 1.533x over V6
   on the tested RTX 3090 workload.

## Conclusion

Implemented, tested, and benchmarked vectorized
CUDA reduction with correct boundary handling,
multi-stage reduction, and memory safety checks.
