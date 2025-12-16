#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

#define KERNEL_SIZE 3
__constant__ float c_Kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void conv2D(float *d_in, float *d_out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return;

    float sum = 0.0f;
    for (int i = -1; i <= 1; ++i) {  // KERNEL_SIZE=3, center at 1
        for (int j = -1; j <= 1; ++j) {
            int curRow = row + i;
            int curCol = col + j;
            if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                sum += d_in[curRow * width + curCol] * c_Kernel[(i+1)*KERNEL_SIZE + (j+1)];
            }
        }
    }
    d_out[row * width + col] = sum;
}

int main() {
    int WIDTH = 1024;
    int HEIGHT = 1024;

    printf("Image size: %dx%d\n", WIDTH, HEIGHT);

    size_t size = WIDTH * HEIGHT * sizeof(float);
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    // Initialize input with 1.0f
    for (size_t i = 0; i < WIDTH * HEIGHT; i++) h_in[i] = 1.0f;

    // Example kernel: Sobel edge detection X (adjust as needed)
    float h_Kernel[KERNEL_SIZE * KERNEL_SIZE] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));

    CHECK_CUDA(cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    conv2D<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, WIDTH, HEIGHT);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU Conv2D Kernel Time: %.6f seconds\n", ms / 1000.0);

    CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Sample output (edges would be detected; for all 1s, sum=0 except boundaries if padded)
    printf("Sample output: h_out[0] = %.1f, h_out[end] = %.1f\n", h_out[0], h_out[WIDTH*HEIGHT - 1]);

    free(h_in); free(h_out);
    CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out));
    return 0;
}
