#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                         \
    do {                                         \
        cudaError_t err = (call);                \
        if (err != cudaSuccess) {                \
            fprintf(stderr, "CUDA Error: %s\n",  \
                    cudaGetErrorString(err));    \
            std::exit(EXIT_FAILURE);             \
        }                                        \
    } while (0)

__global__ void copy_coalesced(
    const float* input,
    float* output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = input[idx];
    }
}

__global__ void copy_strided(
    const float* input,
    float* output,
    int N,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = input[idx * stride];
    }
}

int main() {
    const int N = 1 << 20;
    const int STRIDE = 32;

    const size_t input_count = 
        (size_t)N * STRIDE;

    const size_t input_bytes =
        input_count * sizeof(float);

    const size_t output_bytes =
        (size_t)N * sizeof(float);

    const int block_size = 256;

    const int grid_size = 
        (N + block_size - 1) / block_size;

    float* h_input = (float*)malloc(input_bytes);
    float* h_coalesced_output = (float*)malloc(output_bytes);
    float* h_strided_output = (float*)malloc(output_bytes);

    float *d_input, *d_coalesced_output, *d_strided_output;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input), input_bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_coalesced_output), output_bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_strided_output), output_bytes
    ));

    for(int i = 0; i < N * STRIDE; i++){
        h_input[i] = static_cast<float>(i % 1000);
    }

    CUDA_CHECK(cudaMemcpy(
        d_input, h_input, input_bytes, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());

    copy_coalesced<<<grid_size, block_size>>>(
        d_input, d_coalesced_output, N
    );

    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_coalesced_output, d_coalesced_output, output_bytes, cudaMemcpyDeviceToHost
    ));

    copy_strided<<<grid_size, block_size>>>(
        d_input, d_strided_output, N, STRIDE
    );

    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_strided_output, d_strided_output, output_bytes, cudaMemcpyDeviceToHost
    ));

    bool coalesced_success = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_input[i] - h_coalesced_output[i]) > 1e-5f) {
            printf("Coalesced mismatch at index %d\n", i);
            coalesced_success = false;
            break;
        }
    }

    bool strided_success = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_input[i * STRIDE] - h_strided_output[i]) > 1e-5f) {
            printf("Strided mismatch at index %d\n", i);
            strided_success = false;
            break;
        }
    }

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 5; i++) {
        copy_coalesced<<<grid_size, block_size>>>(
            d_input, d_coalesced_output, N
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);

    for (int i = 0; i < 50; i++) {
        copy_coalesced<<<grid_size, block_size>>>(
            d_input, d_coalesced_output, N
        );
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0.00f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    printf("Coalesced_Total time: %f ms\n", milliseconds);

    printf("Coalesced_Average Time: %f ms\n", milliseconds / 50.0f);

    for (int i = 0; i < 5; i++) {
        copy_strided<<<grid_size, block_size>>>(
            d_input, d_strided_output, N, STRIDE
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);

    for (int i = 0; i < 50; i++) {
        copy_strided<<<grid_size, block_size>>>(
            d_input, d_strided_output, N, STRIDE
        );
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    printf("Strided_Total time: %f ms\n", milliseconds);

    printf("Strided_Average Time: %f ms\n", milliseconds / 50.0f);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_coalesced_output));
    CUDA_CHECK(cudaFree(d_strided_output));

    free(h_input);
    free(h_coalesced_output);
    free(h_strided_output);

    return coalesced_success && strided_success ? EXIT_SUCCESS : EXIT_FAILURE;
}