#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",      \
                    __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                 \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)

__global__ void matrix_add_2d(
    const float* A,
    const float* B,
    float* C,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int rows = 5;
    const int cols = 7;

    const int N = rows * cols;

    const size_t bytes = N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    if(!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_A), bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_B), bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_C), bytes
    ));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A, bytes, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B, bytes, cudaMemcpyHostToDevice
    ));

    dim3 block(4, 2);

    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    printf("Rows = %d, Cols = %d\n", rows, cols);
    printf("Block = (%u, %u)\n", block.x, block.y);
    printf("Grid = (%u, %u)\n", grid.x, grid.y);

    matrix_add_2d<<<grid, block>>>(
        d_A, d_B, d_C, rows, cols
    );

    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_C, d_C, bytes, cudaMemcpyDeviceToHost
    ));

    bool success = true;

    for (int i = 0; i < rows && success; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols +j;
            float expected = h_A[idx] + h_B[idx];

            if (std::fabs(h_C[idx] - expected) > 1e-5f) {
                printf("Mismatch at (%d, %d)\n", i, j);
                success = false;
                break;
            }
        }
    }

    if (success) {
        printf("Verification PASSED!\n");
    } else {
        printf("Verification FAILED!\n");
    }

    for (int i = 0; i < 5; i++) {
        printf("C[0][%d] = %.1f\n", i, h_C[i]);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}