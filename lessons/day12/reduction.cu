#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>

const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;
const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

#define CUDA_CHECK(call) do {                          \
    cudaError_t err = (call);                          \
    if (err != cudaSuccess) {                          \
        std::fprintf(stderr, "CUDA error: %s\n",       \
            cudaGetErrorString(err));                  \
        std::exit(EXIT_FAILURE);                       \
    }                                                  \
} while (0)

__global__ void reduce_v0(
    const float* input,
    float* partial,
    int n
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;

    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;

    __syncthreads();

    for (unsigned int step = 1;
        step < blockDim.x;
        step *= 2) {
        
        if(tid % (2 * step) == 0) {
            smem[tid] += smem[tid + step];
        }
        
        __syncthreads();
    }
    
    if (tid == 0) {
        partial[blockIdx.x] = smem[0];
    }
}

__global__ void reduce_v2(
    const float* input,
    float* partial,
    int n
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;

    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;

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
        partial[blockIdx.x] = smem[0];
    }
}

__global__ void reduce_v3(
    const float* input,
    float* partial,
    int n
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;

    int gid = blockIdx.x * (blockDim.x * 2) + tid;

    float val = 0.0f;

    if (gid < n) {
        val += input[gid];
    }

    if (gid + blockDim.x < n) {
        val += input[gid + blockDim.x];        
    }

    smem[tid] = val;

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
        partial[blockIdx.x] = smem[0];
    }
}

__device__ float warpReduceSum(float val) {
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

__global__ void reduce_v6(
    const float* input,
    float* partial,
    int n
) {
    int tid = threadIdx.x;

    int gid = blockIdx.x * (blockDim.x * 2) + tid;

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
            :0.0f;
        
        val = warpReduceSum(val);
    }

    if (tid == 0) {
        partial[blockIdx.x] = val;
    }
}

__global__ void reduce_final(
    const float* partial,
    float* output,
    int num_partials
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;

    float thread_sum = 0.0f;
    for (int i = tid; i < num_partials; i += blockDim.x) {
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

void launch_v0(
    const float* input,
    float* partial,
    float* output,
    int n,
    int num_blocks
) {
    reduce_v0<<<num_blocks, BLOCK_SIZE,
        BLOCK_SIZE * sizeof(float)>>>(
        input, partial, n
    );

    CUDA_CHECK(cudaGetLastError());

    reduce_final<<<1, BLOCK_SIZE,
        BLOCK_SIZE * sizeof(float)>>>(
        partial, output, num_blocks
    );

    CUDA_CHECK(cudaGetLastError());
}

void launch_v2(
    const float* input,
    float* partial,
    float* output,
    int n,
    int num_blocks
) {
    reduce_v2<<<num_blocks, BLOCK_SIZE,
        BLOCK_SIZE * sizeof(float)>>>(
        input, partial, n
    );

    CUDA_CHECK(cudaGetLastError());

    reduce_final<<<1, BLOCK_SIZE,
        BLOCK_SIZE * sizeof(float)>>>(
        partial, output, num_blocks
    );

    CUDA_CHECK(cudaGetLastError());
}

void launch_v3(
    const float* input,
    float* partial,
    float* output,
    int n,
    int num_blocks
) {
    reduce_v3<<<num_blocks, BLOCK_SIZE,
        BLOCK_SIZE * sizeof(float)>>>(
        input, partial, n
    );

    CUDA_CHECK(cudaGetLastError());

    reduce_final<<<1, BLOCK_SIZE,
        BLOCK_SIZE * sizeof(float)>>>(
        partial, output, num_blocks
    );

    CUDA_CHECK(cudaGetLastError());
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

    reduce_final<<<1, BLOCK_SIZE,
        BLOCK_SIZE * sizeof(float)>>>(
        partial, output, num_blocks
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
"%-4s CPU=%.8f GPU=%.8f Error=%.8e Tol=%.8e [%s]\n",
        name,
        cpu_result,
        gpu_result,
        error,
        tolerance,
        pass ? "PASS" : "FAIL"
    );

    return pass;
}

template <typename LaunchFn>
float benchmark_ms(
    LaunchFn launch,
    int warmup = 10,
    int repeat = 50
) {
    for (int i = 0; i < warmup; ++i) {
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

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms, start, stop
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

int main() {

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::printf("GPU: %s\n", prop.name);

    const int test_sizes[] = {
        1, 31, 32, 33,
        255, 256, 257,
        1000, 65537, 1048576
    };

    std::mt19937 rng(2026);

    std::uniform_real_distribution<float> dist(
        -1.0f, 1.0f
    );

    bool all_pass = true;

    std::printf("\n========== Correctness Tests ==========\n");

    for (int n : test_sizes) {

        std::vector<float> h_input(n);

        double cpu_result = 0.0;

        for (int i = 0; i < n; ++i) {
            h_input[i] = dist(rng);
            cpu_result += static_cast<double>(h_input[i]);
        }

        int blocks_v2 =
            (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        int blocks_v3 = 
            (n + BLOCK_SIZE * 2 - 1)
            / (BLOCK_SIZE * 2);

        float* d_input = nullptr;
        float* d_partial = nullptr;
        float* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(
            (void**)&d_input,
            static_cast<size_t>(n) * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_partial,
            static_cast<size_t>(blocks_v2) * sizeof(float)
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
            "\nN=%d, V2 Blocks=%d, V3/V6 Blocks=%d\n",
            n, blocks_v2, blocks_v3
        );

        auto run0 = [&]() {
            launch_v0(
                d_input, d_partial, d_output,
                n, blocks_v2
            );
        };

        auto run2 = [&]() {
            launch_v2(
                d_input, d_partial, d_output,
                n, blocks_v2
            );
        };

        auto run3 = [&]() {
            launch_v3(
                d_input, d_partial, d_output,
                n, blocks_v3
            );
        };

        auto run6 = [&]() {
            launch_v6(
                d_input, d_partial, d_output,
                n, blocks_v3
            );
        };

        all_pass &= validate(
            "V0", run0, d_output, cpu_result
        );

        all_pass &= validate(
            "V2", run2, d_output, cpu_result
        );

        all_pass &= validate(
            "V3", run3, d_output, cpu_result
        );

        all_pass &= validate(
            "V6", run6, d_output, cpu_result
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

    std::printf("\n========== Performance Benchmark ==========\n");

    const int bench_n = 1 << 24;

    std::vector<float> h_input(bench_n);

    double cpu_result = 0.0;

    for (int i = 0; i < bench_n; ++i) {
        h_input[i] = dist(rng);
        cpu_result += static_cast<double>(h_input[i]);
    }

    int blocks_v2 =
        (bench_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int blocks_v3 =
        (bench_n + BLOCK_SIZE * 2 - 1)
        / (BLOCK_SIZE * 2);

    float* d_input = nullptr;
    float* d_partial = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        (void**)&d_input,
        static_cast<size_t>(bench_n) * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&d_partial,
        static_cast<size_t>(blocks_v2) * sizeof(float)
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

    auto run2 = [&]() {
        launch_v2(
            d_input, d_partial, d_output,
            bench_n, blocks_v2
        );
    };

    auto run3 = [&]() {
        launch_v3(
            d_input, d_partial, d_output,
            bench_n, blocks_v3
        );
    };

    auto run6 = [&]() {
        launch_v6(
            d_input, d_partial, d_output,
            bench_n, blocks_v3
        );
    };

    // Validate before timing
    bool benchmark_correct = true;

    benchmark_correct &= validate(
        "V2", run2, d_output, cpu_result
    );

    benchmark_correct &= validate(
        "V3", run3, d_output, cpu_result
    );

    benchmark_correct &= validate(
        "V6", run6, d_output, cpu_result
    );

    if (!benchmark_correct) {
        std::printf("Benchmark aborted: correctness failed!\n");
        return EXIT_FAILURE;
    }

    float ms_v2 = benchmark_ms(run2);
    float ms_v3 = benchmark_ms(run3);
    float ms_v6 = benchmark_ms(run6);

    auto print_result = [&](const char* name, float ms) {

        double bandwidth =
            static_cast<double>(bench_n) * sizeof(float)
            / (static_cast<double>(ms) * 1e6);

        std::printf(
            "%s: %.5f ms, Input-effective BW: %.2f GB/s\n",
            name,
            ms,
            bandwidth
        );
    };

    print_result("V2", ms_v2);
    print_result("V3", ms_v3);
    print_result("V6", ms_v6);

    std::printf(
        "\nV3 speedup over V2: %.3fx\n",
        ms_v2 / ms_v3
    );

    std::printf(
        "V6 speedup over V2: %.3fx\n",
        ms_v2 / ms_v6
    );

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}