%%writefile vector_mul.cu
#include <stdio.h>
#include <cuda.h>

// CUDA Kernel: Vector multiplication
__global__ void VectorMultiplication(float *d_X, float *d_Y, float *d_resu, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        d_resu[index] = d_X[index] * d_Y[index];
    }
}

int main() {
    const int size = 3;
    const int bytes = size * sizeof(float);

    float h_X[size] = {1.0f, 2.0f, 3.0f};
    float h_Y[size] = {4.0f, 5.0f, 6.0f};
    float h_resu[size] = {0};

    float *d_X, *d_Y, *d_resu;

    cudaMalloc(&d_X, bytes);
    cudaMalloc(&d_Y, bytes);
    cudaMalloc(&d_resu, bytes);

    cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, bytes, cudaMemcpyHostToDevice);

    VectorMultiplication<<<1, size>>>(d_X, d_Y, d_resu, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_resu, d_resu, bytes, cudaMemcpyDeviceToHost);

    printf("Vector X: ");
    for(int i=0;i<size;i++) printf("%.1f ", h_X[i]);
    printf("\n");

    printf("Vector Y: ");
    for(int i=0;i<size;i++) printf("%.1f ", h_Y[i]);
    printf("\n");

    printf("Result (X * Y): ");
    for(int i=0;i<size;i++) printf("%.1f ", h_resu[i]);
    printf("\n");

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_resu);

    return 0;
}
