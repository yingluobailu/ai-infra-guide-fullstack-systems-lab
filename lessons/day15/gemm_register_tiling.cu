
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ===============================================
// Configuration
// ===============================================

// Day 14 baseline
constexpr int TILE = 16;

// Day 15 register tiling
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 8;

constexpr int TM = 2;
constexpr int TN = 2;

constexpr int THREADS = 256;

static_assert(
    BM * BN == THREADS * TM * TN,
    "Thread mapping does not cover the C tile"
);

static_assert(
    BM * BK == THREADS &&
    BK * BN == THREADS,
    "This implementation loads 1 A and 1 B per thread"
);

// ===============================================
// CUDA Error Check
// ===============================================

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        std::fprintf(stderr,                               \
            "CUDA error %s:%d: %s\n",                      \
            __FILE__, __LINE__,                            \
            cudaGetErrorString(err));                       \
        std::exit(EXIT_FAILURE);                           \
    }                                                      \
} while (0)

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// ===============================================
// V1: Day 14 Shared Memory Tiled Baseline
// Each thread computes 1 output.
// ===============================================

__global__ void gemm_shared_tiled(
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

    for (int k0 = 0; k0 < K; k0 += TILE) {

        int a_col = k0 + tx;
        int b_row = k0 + ty;

        As[ty][tx] =
            (row < M && a_col < K)
            ? A[row * K + a_col]
            : 0.0f;

        Bs[ty][tx] =
            (b_row < K && col < N)
            ? B[b_row * N + col]
            : 0.0f;

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

// ===============================================
// V2: Thread Tiling + Register Tiling
// Each thread computes a 2x2 output tile.
// ===============================================

__global__ void gemm_register_tiled(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;

    // -------------------------------------------
    // 1. Each thread owns a 2x2 C tile
    // -------------------------------------------

    constexpr int THREAD_COLS = BN / TN;

    int thread_row = (tid / THREAD_COLS) * TM;
    int thread_col = (tid % THREAD_COLS) * TN;

    // Thread-private accumulators
    float acc[TM][TN] = {};

    // -------------------------------------------
    // 2. Traverse K dimension in BK-sized tiles
    // -------------------------------------------

    for (int k0 = 0; k0 < K; k0 += BK) {

        // ---------------------------------------
        // 3. Cooperative load: A -> Shared Memory
        // ---------------------------------------

        int a_row = tid / BK;
        int a_col = tid % BK;

        int global_a_row =
            blockIdx.y * BM + a_row;

        int global_a_col = k0 + a_col;

        As[a_row][a_col] =
            (global_a_row < M && global_a_col < K)
            ? A[global_a_row * K + global_a_col]
            : 0.0f;

        // ---------------------------------------
        // 4. Cooperative load: B -> Shared Memory
        // ---------------------------------------

        int b_row = tid / BN;
        int b_col = tid % BN;

        int global_b_row = k0 + b_row;

        int global_b_col =
            blockIdx.x * BN + b_col;

        Bs[b_row][b_col] =
            (global_b_row < K && global_b_col < N)
            ? B[global_b_row * N + global_b_col]
            : 0.0f;

        // All threads wait for A/B tiles.
        __syncthreads();

        // ---------------------------------------
        // 5. Shared Memory -> Register fragments
        //    Compute outer products
        // ---------------------------------------

        #pragma unroll
        for (int k = 0; k < BK; ++k) {

            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] =
                    As[thread_row + i][k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] =
                    Bs[k][thread_col + j];
            }

            // Register outer product
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] +=
                        a_frag[i] * b_frag[j];
                }
            }
        }

        // Prevent overwrite before all reads end.
        __syncthreads();
    }

    // -------------------------------------------
    // 6. Write the 2x2 output tile
    // -------------------------------------------

    int row0 =
        blockIdx.y * BM + thread_row;

    int col0 =
        blockIdx.x * BN + thread_col;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {

            int row = row0 + i;
            int col = col0 + j;

            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

// ===============================================
// Kernel Launch Wrappers
// ===============================================

enum class Version {
    SharedTiled,
    RegisterTiled
};

void launch_gemm(
    Version version,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    if (version == Version::SharedTiled) {

        dim3 block(TILE, TILE);

        dim3 grid(
            ceil_div(N, TILE),
            ceil_div(M, TILE)
        );

        gemm_shared_tiled<<<grid, block>>>(
            A, B, C, M, N, K
        );

    } else {

        dim3 block(THREADS);

        dim3 grid(
            ceil_div(N, BN),
            ceil_div(M, BM)
        );

        gemm_register_tiled<<<grid, block>>>(
            A, B, C, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

// ===============================================
// CPU Reference: double precision accumulation
// ===============================================

void cpu_gemm(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<double>& C,
    int M, int N, int K
) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {

            double sum = 0.0;

            for (int k = 0; k < K; ++k) {
                sum +=
                    static_cast<double>(
                        A[row * K + k]
                    ) *
                    static_cast<double>(
                        B[k * N + col]
                    );
            }

            C[row * N + col] = sum;
        }
    }
}

// ===============================================
// Correctness Validation
// ===============================================

bool validate(
    const char* name,
    const std::vector<float>& gpu,
    const std::vector<double>& cpu
) {
    int failures = 0;
    double max_error = 0.0;

    for (size_t i = 0; i < gpu.size(); ++i) {

        double got = static_cast<double>(gpu[i]);
        double expected = cpu[i];

        double error = std::abs(got - expected);

        double tolerance =
            1e-3 + 1e-3 * std::abs(expected);

        if (!std::isfinite(got) ||
            error > tolerance) {

            if (failures < 3) {
                std::printf(
                    "Mismatch index=%zu "
                    "GPU=%.8f CPU=%.8f\n",
                    i, got, expected
                );
            }

            ++failures;
        }

        max_error = std::max(max_error, error);
    }

    bool pass = (failures == 0);

    std::printf(
        "%-14s MaxError=%.8e Failures=%d [%s]\n",
        name,
        max_error,
        failures,
        pass ? "PASS" : "FAIL"
    );

    return pass;
}

// ===============================================
// CUDA Event Benchmark
// ===============================================

float benchmark_ms(
    Version version,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int warmup = 10,
    int repeat = 50
) {
    for (int i = 0; i < warmup; ++i) {
        launch_gemm(
            version, A, B, C, M, N, K
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        launch_gemm(
            version, A, B, C, M, N, K
        );
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

// ===============================================
// Run One Test Case
// ===============================================

bool run_case(
    int M, int N, int K,
    bool do_benchmark
) {
    std::printf(
        "\n===== M=%d N=%d K=%d =====\n",
        M, N, K
    );

    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C(size_C);

    std::vector<double> h_ref(size_C, 0.0);

    std::mt19937 rng(2026);

    std::uniform_real_distribution<float> dist(
        -1.0f, 1.0f
    );

    for (float& x : h_A) {
        x = dist(rng);
    }

    for (float& x : h_B) {
        x = dist(rng);
    }

    cpu_gemm(h_A, h_B, h_ref, M, N, K);

    // -------------------------------------------
    // GPU Memory Allocation
    // -------------------------------------------

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(
        (void**)&d_A,
        size_A * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&d_B,
        size_B * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&d_C,
        size_C * sizeof(float)
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

    // -------------------------------------------
    // V1: Shared Memory Baseline
    // -------------------------------------------

    launch_gemm(
        Version::SharedTiled,
        d_A, d_B, d_C, M, N, K
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C,
        size_C * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool pass_shared = validate(
        "SharedTiled", h_C, h_ref
    );

    // -------------------------------------------
    // V2: Register Tiled GEMM
    // -------------------------------------------

    launch_gemm(
        Version::RegisterTiled,
        d_A, d_B, d_C, M, N, K
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C,
        size_C * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool pass_register = validate(
        "RegisterTiled", h_C, h_ref
    );

    bool pass = pass_shared && pass_register;

    // -------------------------------------------
    // Performance Benchmark
    // -------------------------------------------

    if (pass && do_benchmark) {

        float ms_shared = benchmark_ms(
            Version::SharedTiled,
            d_A, d_B, d_C, M, N, K
        );

        float ms_register = benchmark_ms(
            Version::RegisterTiled,
            d_A, d_B, d_C, M, N, K
        );

        double flops =
            2.0 * static_cast<double>(M)
                * static_cast<double>(N)
                * static_cast<double>(K);

        auto tflops = [&](float ms) {
            return flops /
                (static_cast<double>(ms) * 1e9);
        };

        std::printf(
            "SharedTiled:   %.6f ms, %.4f TFLOPS\n",
            ms_shared,
            tflops(ms_shared)
        );

        std::printf(
            "RegisterTiled: %.6f ms, %.4f TFLOPS\n",
            ms_register,
            tflops(ms_register)
        );

        std::printf(
            "Register speedup: %.3fx\n",
            ms_shared / ms_register
        );
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return pass;
}

// ===============================================
// Main
// ===============================================

int main(int argc, char** argv) {

    // --check-only skips performance measurement
    // to make compute-sanitizer faster.
    bool check_only =
        (argc > 1 &&
         std::strcmp(argv[1], "--check-only") == 0);

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::printf("GPU: %s\n", prop.name);

    std::printf(
        "BM=%d BN=%d BK=%d TM=%d TN=%d\n",
        BM, BN, BK, TM, TN
    );

    bool all_pass = true;

    // Small and irregular matrices
    all_pass &= run_case(1, 1, 1, false);
    all_pass &= run_case(2, 2, 3, false);
    all_pass &= run_case(17, 19, 23, false);
    all_pass &= run_case(31, 33, 7, false);
    all_pass &= run_case(35, 50, 30, false);
    all_pass &= run_case(63, 65, 31, false);
    all_pass &= run_case(128, 96, 80, false);

    // Larger matrix
    all_pass &= run_case(
        512, 512, 512, !check_only
    );

    std::printf(
        "\nOverall: %s\n",
        all_pass ? "ALL PASS" : "SOME FAILED"
    );

    return all_pass
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
