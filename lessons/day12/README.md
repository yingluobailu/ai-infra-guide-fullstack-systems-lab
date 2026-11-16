# Day 12 — CUDA Parallel Reduction

## 1. Learning Objectives

Implement and optimize CUDA sum reduction.

Key concepts:
- Shared Memory tree reduction
- Warp divergence and bank conflicts
- Multiple elements per thread
- Warp Shuffle
- Two-stage reduction
- CUDA Event benchmarking

## 2. Implementations

| Version | Description |
|---------|-------------|
| V0 | Naive interleaved reduction |
| V2 | Sequential addressing reduction |
| V3 | Two elements per thread |
| V6 | Warp Shuffle reduction |

All versions use a second-stage `reduce_final` kernel
to combine block-level partial sums.

## 3. Environment

- GPU: NVIDIA GeForce RTX 3090
- CUDA architecture: sm_86
- Block size: 256 threads
- Language: CUDA C++17

## 4. Build and Run

```bash
mkdir -p build/day12

nvcc -std=c++17 -O3 -arch=sm_86 \
    lessons/day12/reduction.cu \
    -o build/day12/reduction

./build/day12/reduction
```

Memory checking:

```bash
compute-sanitizer --tool memcheck \
    ./build/day12/reduction
```

## 5. Correctness

Test sizes:

1, 31, 32, 33, 255, 256, 257,
1000, 65537, 1048576

All four implementations passed.

CPU reference uses double accumulation.

Tolerance:
`1e-3 + 1e-5 * abs(cpu_result)`

Compute Sanitizer:
`ERROR SUMMARY: 0 errors`

## 6. Performance Benchmark

GPU: NVIDIA GeForce RTX 3090

Input size: 16,777,216 float32 elements

Warmup: 10 iterations

Measurement: 50 iterations, CUDA Events

Timing includes both reduction stages.

| Version | Time (ms) | Effective BW (GB/s) |
|---------|-----------|---------------------|
| V2 | 0.31117 | 215.66 |
| V3 | 0.16220 | 413.74 |
| V6 | 0.12554 | 534.55 |

Speedup over V2:
- V3: 1.918x
- V6: 2.479x

Bandwidth is estimated from input bytes only,
not total measured DRAM traffic.

## 7. Key Findings

1. V2 reduces divergence and improves shared
   memory access patterns compared with V0.
2. V3 processes two elements per thread,
   reducing the number of blocks and partial sums.
3. V6 uses Warp Shuffle for warp-level reduction
   and Shared Memory for inter-warp communication.
4. Warp Shuffle reduces synchronization overhead,
   but performance improvements depend on workload
   size and hardware characteristics.
5. Performance tests under Compute Sanitizer
   must not be compared with normal GPU timings.

## 8. Conclusion

Implemented and validated four CUDA reduction
kernels with boundary handling, CUDA error checks,
two-stage reduction, and performance benchmarks.
