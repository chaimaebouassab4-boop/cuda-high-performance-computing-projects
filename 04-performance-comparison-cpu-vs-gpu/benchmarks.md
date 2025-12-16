## cpu_version.c

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int N;
    printf("Enter matrix size N: ");
    scanf("%d", &N);

    float *A = malloc(N*N*sizeof(float));
    float *B = malloc(N*N*sizeof(float));
    float *C = malloc(N*N*sizeof(float));

    for (int i = 0; i < N*N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    clock_t start = clock();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i*N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("CPU Execution Time: %f seconds\n", time_spent);

    free(A); free(B); free(C);
    return 0;
}
```

---

## gpu_version.cu

```c
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matMul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row*N + k] * B[k*N + col];
        }
        C[row*N + col] = sum;
    }
}

int main() {
    int N;
    printf("Enter matrix size N: ");
    scanf("%d", &N);

    int bytes = N*N*sizeof(float);

    float *hA = malloc(bytes);
    float *hB = malloc(bytes);
    float *hC = malloc(bytes);

    for (int i = 0; i < N*N; i++) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((N+15)/16, (N+15)/16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMul<<<blocks, threads>>>(dA, dB, dC, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Execution Time: %f ms\n", milliseconds);

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);

    return 0;
}
```

---

## benchmarks.md

```md
# CPU vs GPU Performance Comparison

## Objective
Compare execution time of matrix multiplication on CPU and GPU.

## Methodology
- Same matrix size N
- Same input values
- CPU uses triple nested loop
- GPU uses CUDA kernel (2D grid)

## Metrics
- CPU time measured using `clock()`
- GPU time measured using `cudaEvent`

## Expected Results
- For small N: CPU may be faster
- For large N: GPU significantly faster

## Conclusion
GPU parallelism provides major performance improvement for large matrices.
```

---

## charts.png

```
[Placeholder]
This image should contain a bar or line chart comparing CPU vs GPU execution time
for different matrix sizes (e.g., N = 128, 256, 512, 1024).
```
