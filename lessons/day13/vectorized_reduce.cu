#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstddef>


constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

static_assert(BLOCK_SIZE % WARP_SIZE == 0,
              "BLOCK_SIZE must be divisible by WARP_SIZE");

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",   \
            __FILE__, __LINE__, cudaGetErrorString(err));  \
        std::exit(EXIT_FAILURE);                           \
    }                                                      \
} while (0)

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__
float warpReduceSum(float val) {
    for (int offset = 16;
        offset > 0;
        offset >>= 1) {
        
        val += __shfl_down_sync(
            0xffffffff,
            val,
            offset
        );

    }

    return val;
}

__global__ void reduce_final(
    const float* partial,
    float* output,
    int num_partials
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;

    float thread_sum = 0.0f;

    for (int i = tid;
        i < num_partials;
        i += blockDim.x) {
        
        thread_sum += partial[i];
    }

    smem[tid] = thread_sum;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2;
        s > 0;
        s >>= 1) {
        
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        *output = smem[0];
    }
}

__global__ void reduce_v6(
    const float* input,
    float* partial,
    int n
) {
    int tid = threadIdx.x;

    int gid =
        blockIdx.x * (blockDim.x * 2) + tid;

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    float val = 0.0f;

    if (gid < n) {
        val += input[gid];
    }

    if (gid + blockDim.x < n) {
        val += input[gid + blockDim.x];
    }

    val = warpReduceSum(val);

    __shared__ float warp_results[NUM_WARPS];

    if (lane == 0) {
        warp_results[wid] = val;
    }

    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;

    if (wid == 0) {
        val = (lane < num_warps)
            ? warp_results[lane]
            : 0.0f;

        val = warpReduceSum(val);
    }

    if (tid == 0) {
        partial[blockIdx.x] = val;
    }
}

__global__ void reduce_v7(
    const float* input,
    float* partial,
    int n
) {
    int tid = threadIdx.x;

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    const float4* input4 = 
        reinterpret_cast<const float4*>(input);

    int n4 = n / 4;

    float val = 0.0f;

    int global_tid = 
        blockIdx.x * blockDim.x + tid;

    int stride = 
        gridDim.x * blockDim.x;

    for (int idx = global_tid;
        idx < n4;
        idx += stride) {
        float4 data = input4[idx];

        val += data.x;
        val += data.y;
        val += data.z;
        val += data.w;
    }

    int tail_start = n4 * 4;

    for (int idx = tail_start + global_tid;
        idx < n;
        idx += stride) {
        val += input[idx];
    }

    val = warpReduceSum(val);

    __shared__ float warp_results[NUM_WARPS];

    if (lane == 0) {
        warp_results[wid] = val;
    }

    __syncthreads();

    if (wid == 0) {
        val = (lane < NUM_WARPS)
            ? warp_results[lane]
            : 0.0f;

        val = warpReduceSum(val);
    }

    if (tid == 0) {
        partial[blockIdx.x] = val;
    }
}

void launch_v6(
    const float* input,
    float* partial,
    float* output,
    int n,
    int num_blocks
) {
    reduce_v6<<<num_blocks, BLOCK_SIZE>>>(
        input, partial, n
    );

    CUDA_CHECK(cudaGetLastError());

    reduce_final<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        partial, output, num_blocks
    );

    CUDA_CHECK(cudaGetLastError());
}

void launch_v7(
    const float* input,
    float* partial,
    float* output,
    int n,
    int grid_size
) {
    reduce_v7<<<grid_size, BLOCK_SIZE>>>(
        input, partial, n
    );

    CUDA_CHECK(cudaGetLastError());

    reduce_final<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        partial, output, grid_size
    );

    CUDA_CHECK(cudaGetLastError());
}

template <typename LaunchFn>
bool validate(
    const char* name,
    LaunchFn launch,
    float* d_output,
    double cpu_result
) {
    launch();

    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_result = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &gpu_result,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    double error = std::fabs(
        static_cast<double>(gpu_result) - cpu_result
    );

    double tolerance =
        1e-3 + 1e-5 * std::fabs(cpu_result);
    
    bool pass = error <= tolerance;

    std::printf(
        "%-4s CPU=%.8f GPU=%.8f "
        "Error=%.8e Tol=%.8e [%s]\n",
        name,
        cpu_result,
        gpu_result,
        error,
        tolerance,
        pass ? "Pass" : "FAIL"        
    );

    return pass;
}

template <typename LaunchFn>
float benchmark_ms(
    LaunchFn launch,
    int warmup = 10,
    int repeat = 50
) {
    for (int i = 0; i < warmup; i++) {
        launch();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        launch();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int num_sms = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(
        &num_sms,
        cudaDevAttrMultiProcessorCount,
        0
    ));

    int target_grid = num_sms * 4;

    std::printf("GPU: %s\n", prop.name);
    std::printf("SM count: %d\n", num_sms);
    std::printf("V7 target grid: %d\n", target_grid);

    std::mt19937 rng(2026);

    std::uniform_real_distribution<float> dist(
        -1.0f, 1.0f
    );

    const int test_sizes[] = {
        1, 3, 4, 5, 7,
        31, 32, 33,
        255, 256, 257,
        1000, 65537, 1048576
    };

    bool all_pass = true;

    std::printf(
        "\n========== Correctness Tests ==========\n"
    );

    for (int n : test_sizes) {
        std::vector<float> h_input(n);

        double cpu_result = 0.0;

        for (int i = 0; i < n; ++i) {
            h_input[i] = dist(rng);

            cpu_result += 
                static_cast<double>(h_input[i]);
        }

        int blocks_v6 = 
            ceil_div(n, BLOCK_SIZE * 2);

        int blocks_v7 = std::min(
            target_grid,
            std::max(1, ceil_div(n, BLOCK_SIZE * 4))
        );

        int partial_capacity =
            std::max(blocks_v6, blocks_v7);

        float* d_input = nullptr;
        float* d_partial = nullptr;
        float* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(
            (void**)&d_input,
            static_cast<size_t>(n) * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_partial,
            static_cast<size_t>(partial_capacity) * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_output,
            sizeof(float)
        ));

        CUDA_CHECK(cudaMemcpy(
            d_input,
            h_input.data(),
            static_cast<size_t>(n) * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        std::printf(
            "\nN=%d, V6 Blocks=%d, V7 Blocks=%d\n",
            n, blocks_v6, blocks_v7
        );

        auto run6 = [&]() {
            launch_v6(d_input, d_partial, d_output, n, blocks_v6);
        };

        auto run7 = [&]() {
            launch_v7(d_input, d_partial, d_output, n, blocks_v7);
        };

        all_pass &= validate(
            "V6", run6, d_output, cpu_result
        );

        all_pass &= validate(
            "V7", run7, d_output, cpu_result
        );

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaFree(d_output));
    }

    std::printf(
        "\nCorrectness: %s\n",
        all_pass ? "ALL PASS" : "FAILED"
    );

    if (!all_pass) {
        return EXIT_FAILURE;
    }

    std::printf(
        "\n========== Performance Benchmark ==========\n"
    );

    const int bench_n = 1 << 27;

    std::vector<float> h_input(bench_n);

    double cpu_result = 0.0;

    for (int i = 0; i < bench_n; ++i) {
        h_input[i] = dist(rng);

        cpu_result += static_cast<double>(h_input[i]);
    }

    int blocks_v6 = ceil_div(bench_n, BLOCK_SIZE * 2);

    int blocks_v7 = target_grid;

    int partial_capacity = std::max(blocks_v6, blocks_v7);

    float* d_input = nullptr;
    float* d_partial = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        (void**)&d_input,
        static_cast<size_t>(bench_n) * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&d_partial,
        static_cast<size_t>(partial_capacity) * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&d_output,
        sizeof(float)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        static_cast<size_t>(bench_n) * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    auto run6 = [&]() {
        launch_v6(d_input, d_partial, d_output, bench_n, blocks_v6);
    };

    auto run7 = [&]() {
        launch_v7(d_input, d_partial, d_output, bench_n, blocks_v7);
    };

    bool benchmark_correct = true;

    benchmark_correct &= validate(
        "V6", run6, d_output, cpu_result
    );

    benchmark_correct &= validate(
        "V7", run7, d_output, cpu_result
    );

    if (!benchmark_correct) {
        std::printf(
            "Benchmark aborted: correctness failed!\n"
        );

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaFree(d_output));

        return EXIT_FAILURE;
    }

    float ms_v6 = benchmark_ms(run6);
    float ms_v7 = benchmark_ms(run7);

    auto print_result = [&](const char* name, float ms) {

        double bandwidth =
            static_cast<double>(bench_n)
            * sizeof(float)
            / (static_cast<double>(ms) * 1e6);

        std::printf(
            "%s: %.5f ms, "
            "Input-effective BW: %.2f GB/s\n",
            name,
            ms,
            bandwidth
        );
    };

    print_result("V6", ms_v6);
    print_result("V7", ms_v7);

    std::printf(
        "\nV7 speedup over V6: %.3fx\n",
        ms_v6 / ms_v7
    );

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}