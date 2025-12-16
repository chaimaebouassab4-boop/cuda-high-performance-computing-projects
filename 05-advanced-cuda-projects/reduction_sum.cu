#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

__global__ void reduceSum(float *d_in, float *d_out, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;  // For tree reduction handling 2 elements per thread

    // Load data (handle odd sizes)
    sdata[tid] = (idx < n) ? d_in[idx] : 0.0f;
    if (idx + blockDim.x < n) sdata[tid] += d_in[idx + blockDim.x];
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block sum to output
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

int main() {
    int N = 1 << 20;  // 1M elements

    printf("Array size N = %d\n", N);

    size_t size = N * sizeof(float);
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(sizeof(float));  // Final sum

    // Initialize with 1.0f (expected sum = N)
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, size));
    int numBlocks = (N + 1023) / 1024;  // For 1024 threads/block, adjust as needed
    CHECK_CUDA(cudaMalloc(&d_out, numBlocks * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    reduceSum<<<numBlocks, 1024 / 2, 1024 / 2 * sizeof(float)>>>(d_in, d_out, N);  // 512 threads to handle 1024 elements/block
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU Reduction Kernel Time: %.6f seconds\n", ms / 1000.0);

    // Sum block results on host (for simplicity; could do multi-pass for large numBlocks)
    float *h_block_sums = (float*)malloc(numBlocks * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_block_sums, d_out, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    *h_out = 0.0f;
    for (int i = 0; i < numBlocks; i++) *h_out += h_block_sums[i];

    printf("Computed sum = %.0f (Expected: %d)\n", *h_out, N);

    free(h_in); free(h_out); free(h_block_sums);
    CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out));
    return 0;
}
