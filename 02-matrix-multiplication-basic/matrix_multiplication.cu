#include <stdio.h>
#include <cuda_runtime.h>

#define N 2  // Matrix size (2x2)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void MatrixMultiplication(float *d_A, float *d_B, float *d_C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += d_A[row * n + k] * d_B[k * n + col];
        }
        d_C[row * n + col] = sum;
    }
}

int main() {
    int size = N * N;
    int bytes = size * sizeof(float);
    
    float h_A[N][N] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    float h_B[N][N] = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    float h_C[N][N] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    
    float *d_A, *d_B, *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(1, 1);
    
    MatrixMultiplication<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    
    printf("Matrice A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f ", h_A[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrice B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f ", h_B[i][j]);
        }
        printf("\n");
    }
    
    printf("\nResultat C = A x B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f ", h_C[i][j]);
        }
        printf("\n");
    }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
