#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(call) do {                          \
    cudaError_t err = (call);                          \
    if (err != cudaSuccess) {                          \
        std::fprintf(stderr, "CUDA error: %s\n",       \
            cudaGetErrorString(err));                  \
        std::exit(EXIT_FAILURE);                       \
    }                                                  \
} while (0)


constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;


__global__ void transpose_naive(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        int row = y + j;

        if (x < cols && row < rows) {
            output[x * rows + row] =
                input[row * cols + x];
        }
    }
}


template<int PAD>
__global__ void transpose_tiled(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    __shared__ float tile[TILE][TILE + PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;

    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if(x < cols && y + j < rows) {
            tile[ty + j][tx] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    int out_col = blockIdx.y * TILE + tx;
    int out_row = blockIdx.x * TILE + ty;

    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if(out_row + j < cols && out_col < rows) {
            output[(out_row + j) * rows + out_col] = tile[tx][ty + j]; 
        }
    }
}


void launch_kernel(
    int mode,
    dim3 grid,
    dim3 block,
    const float* d_input,
    float* d_output,
    int rows,
    int cols
) {
    if (mode == 0) {
        transpose_naive<<<grid, block>>>(
            d_input, d_output, rows, cols
        );
    } else if (mode == 1) {
        transpose_tiled<0><<<grid, block>>>(
            d_input, d_output, rows, cols
        );
    } else {
        transpose_tiled<1><<<grid, block>>>(
            d_input, d_output, rows, cols
        );
    }
}


int main() {
    const int rows = 1019;
    const int cols = 1507;

    const size_t count = (size_t)rows * cols;
    const size_t bytes = count * sizeof(float);

    std::vector<float> h_input(count);
    std::vector<float> h_output(count);

    for (size_t i = 0; i < count; ++i) {
        h_input[i] = static_cast<float>(i % 997);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input), bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_output), bytes
    ));

    CUDA_CHECK(cudaMemcpy(
        d_input, h_input.data(),
        bytes, cudaMemcpyHostToDevice
    ));

    dim3 block(TILE, BLOCK_ROWS);

    dim3 grid(
        (cols + TILE - 1) / TILE,
        (rows + TILE - 1) / TILE
    );

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const char* names[] = {
        "Naive",
        "Shared_No_Pad",
        "Shared_Padded"
    };

    const int warmup = 5;
    const int repeat = 50;
    
    bool all_passed = true;

    for (int mode = 0; mode < 3; ++mode) {
        CUDA_CHECK(cudaMemset(d_output, 0xFF, bytes));

        launch_kernel(
            mode, grid, block,
            d_input, d_output, rows, cols
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            h_output.data(), d_output,
            bytes, cudaMemcpyDeviceToHost
        ));

        bool passed = true;

        for (int r = 0; r < rows && passed; ++ r) {
            for (int c = 0; c < cols; ++ c) {
                size_t input_idx =
                    (size_t)r * cols + c;
                size_t output_idx =
                    (size_t)c * rows + r;

                float expected = h_input[input_idx];
                float actual = h_output[output_idx];

                if (!std::isfinite(actual) ||
                    actual != expected) {

                    std::printf(
                        "%s FAILED at (%d, %d)\n",
                        names[mode], r, c
                    );

                    passed = false;
                    break;
                }
            }
        }

        if (!passed) {
            all_passed = false;
            break;
        }

        for (int i = 0; i < warmup; ++i) {
            launch_kernel(
                mode, grid, block,
                d_input, d_output, rows, cols
            );
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i) {
            launch_kernel(
                mode, grid, block,
                d_input, d_output, rows, cols
            );
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &total_ms, start, stop
        ));

        float avg_ms = total_ms / repeat;

        double effective_gbps =
            (2.0 * bytes) / (avg_ms * 1e6);

        std:: printf(
            "%s: avg=%.6f ms, BW=%.2f GB/S\n",
            names[mode], avg_ms, effective_gbps
        );

        if (!passed) {
            all_passed = false;
            break;
        }

        std::printf("%s Verification PASSED!\n", names[mode]);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}