%%writefile vector_mul.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void VectorMultiplication(float *d_X, float *d_Y, float *d_resu, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        d_resu[index] = d_X[index] * d_Y[index];
    }
}

int main() {
    const int size = 3;
    const int bytes = size * sizeof(float);
    float h_X[] = {1.0f, 2.0f, 3.0f};
    float h_Y[] = {4.0f, 5.0f, 6.0f};
    float h_resu[3] = {0};
    float *d_X, *d_Y, *d_resu;
    
    CUDA_CHECK(cudaMalloc(&d_X, bytes));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes));
    CUDA_CHECK(cudaMalloc(&d_resu, bytes));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, bytes, cudaMemcpyHostToDevice));
    
    VectorMultiplication<<<1, size>>>(d_X, d_Y, d_resu, size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_resu, d_resu, bytes, cudaMemcpyDeviceToHost));
    
    printf("X: %.1f %.1f %.1f\n", h_X[0], h_X[1], h_X[2]);
    printf("Y: %.1f %.1f %.1f\n", h_Y[0], h_Y[1], h_Y[2]);
    printf("Resultat: %.1f %.1f %.1f\n", h_resu[0], h_resu[1], h_resu[2]);
    
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_resu);
    return 0;
}