#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int TILE = 16;

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        std::fprintf(stderr,                               \
            "CUDA ERROR %s:%d: %s\n",                      \
            __FILE__, __LINE__,                            \
            cudaGetErrorString(err));                       \
        std::exit(EXIT_FAILURE);                           \
    }                                                      \
} while (0)

__global__ void gemm_naive(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

__global__ void gemm_tiled(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + tx;

        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        int b_row = t * TILE + ty;

        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void cpu_gemm(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<double>& C,
    int M, int N, int K
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;

            for (int k = 0; k < K; ++k) {
                sum += static_cast<double>(A[i * K + k]) 
                    * static_cast<double>(B[k * N + j]);
            }

            C[i * N + j] = sum;
        }
    }
}

bool validate(
    const char* name,
    const std::vector<float>& gpu,
    const std::vector<double>& cpu
) {
    double max_error = 0.0;
    int failures = 0;

    for (size_t i = 0; i < gpu.size(); ++i) {

        double got = static_cast<double>(gpu[i]);
        double expected = cpu[i];

        double error = std::abs(got - expected);

        double tolerance =
            1e-3 + 1e-3 * std::abs(expected);

        max_error = std::max(max_error, error);

        if (!std::isfinite(got) ||
            error > tolerance) {
            if (failures < 3) {
                std::printf(
                    "  mismatch at index %zu: "
                    "GPU=%.8f CPU=%.8f\n",
                    i, got, expected
                );
            }
            ++failures;
        }
    }

    bool pass = (failures == 0);

    std::printf(
        "%s: max_error=%.8e failures=%d [%s]\n",
        name,
        max_error,
        failures,
        pass ? "PASS" : "FAIL"
    );

    return pass;
}

// ====================================================
// Kernel launch helper
// ====================================================

enum class Version {
    Naive,
    Tiled
};

void launch_gemm(
    Version version,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    dim3 block(TILE, TILE);

    dim3 grid(
        ceil_div(N, TILE),
        ceil_div(M, TILE)
    );

    if (version == Version::Naive) {
        gemm_naive<<<grid, block>>>(
            A, B, C, M, N, K
        );
    } else {
        gemm_tiled<<<grid, block>>>(
            A, B, C, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

// ====================================================
// CUDA Event Benchmark
// ====================================================

float benchmark_ms(
    Version version,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    int warmup = 10,
    int repeat = 50
) {
    for (int i = 0; i < warmup; ++i) {
        launch_gemm(version, A, B, C, M, N, K);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        launch_gemm(version, A, B, C, M, N, K);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms, start, stop
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

// ====================================================
// Run one test case
// ====================================================

bool run_case(
    int M,
    int N,
    int K,
    bool do_benchmark
) {
    std::printf(
        "\n===== M=%d N=%d K=%d =====\n",
        M, N, K
    );

    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    // --------------------------------------------
    // Prepare input data
    // --------------------------------------------

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);

    std::vector<double> h_ref(size_C, 0.0);
    std::vector<float> h_out(size_C, 0.0f);

    std::mt19937 rng(2026);

    std::uniform_real_distribution<float> dist(
        -1.0f, 1.0f
    );

    for (float& v : h_A) {
        v = dist(rng);
    }

    for (float& v : h_B) {
        v = dist(rng);
    }

    cpu_gemm(h_A, h_B, h_ref, M, N, K);

    // --------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(
        (void**)&d_A, size_A * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&d_B, size_B * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&d_C, size_C * sizeof(float)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(),
        size_A * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(),
        size_B * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // --------------------------------------------
    // Test V0: Naive GEMM
    // --------------------------------------------

    launch_gemm(
        Version::Naive,
        d_A, d_B, d_C, M, N, K
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_out.data(),
        d_C,
        size_C * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool pass_naive = validate(
        "Naive", h_out, h_ref
    );

    // --------------------------------------------
    // Test V1: Shared Tiled GEMM
    // --------------------------------------------

    launch_gemm(
        Version::Tiled,
        d_A, d_B, d_C, M, N, K
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_out.data(),
        d_C,
        size_C * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool pass_tiled = validate(
        "Tiled", h_out, h_ref
    );

    bool pass = pass_naive && pass_tiled;

    // --------------------------------------------
    // Benchmark only if both are correct
    // --------------------------------------------

    if (pass && do_benchmark) {

        float ms_naive = benchmark_ms(
            Version::Naive,
            d_A, d_B, d_C, M, N, K
        );

        float ms_tiled = benchmark_ms(
            Version::Tiled,
            d_A, d_B, d_C, M, N, K
        );

        double flops =
            2.0 * static_cast<double>(M)
                * static_cast<double>(N)
                * static_cast<double>(K);

        // ms -> seconds; report TFLOPS
        auto tflops = [&](float ms) {
            return flops /
                (static_cast<double>(ms) * 1e9);
        };

        std::printf(
            "Naive: %.6f ms, %.4f TFLOPS\n",
            ms_naive,
            tflops(ms_naive)
        );

        std::printf(
            "Tiled: %.6f ms, %.4f TFLOPS\n",
            ms_tiled,
            tflops(ms_tiled)
        );

        std::printf(
            "Tiled / Naive speedup: %.3fx\n",
            ms_naive / ms_tiled
        );
    }

    // --------------------------------------------
    // Release GPU memory
    // --------------------------------------------

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return pass;
}

// ====================================================
// Main
// ====================================================

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::printf("GPU: %s\n", prop.name);
    std::printf("GEMM TILE: %d\n", TILE);

    bool all_pass = true;

    // Tiny matrix
    all_pass &= run_case(1, 1, 1, false);

    // A[2,3] x B[3,2]
    all_pass &= run_case(2, 2, 3, false);

    // Irregular dimensions
    all_pass &= run_case(17, 19, 23, false);
    all_pass &= run_case(63, 65, 31, false);

    // Rectangular matrix
    all_pass &= run_case(128, 96, 80, false);

    // Larger square matrix + Benchmark
    all_pass &= run_case(512, 512, 512, true);

    std::printf(
        "\nOverall: %s\n",
        all_pass ? "ALL PASS" : "SOME FAILED"
    );

    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}