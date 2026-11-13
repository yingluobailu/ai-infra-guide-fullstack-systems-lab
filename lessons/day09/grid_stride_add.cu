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

__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1003;
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

    const int block_size = 128;

    const int grid_size = 2;

    printf("N = %d\n", N);
    printf("Block Size = %d\n", block_size);
    printf("Grid Size = %d\n", grid_size);
    printf("Total Threads = %d\n",
           grid_size * block_size);

    vector_add<<<grid_size, block_size>>>(
        d_A, d_B, d_C, N
    );

    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_C, d_C, bytes, cudaMemcpyDeviceToHost
    ));

    bool success = true;

    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];

        if (std::fabs(h_C[i] - expected) > 1e-5f) {
            printf("Mismatch at index %d\n", i);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Verification PASSED!\n");
    } else {
        printf("Verification FAILED!\n");
    }

    for (int i = 0; i < 5; i++) {
        printf("C[%d] = %.1f\n", i, h_C[i]);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}