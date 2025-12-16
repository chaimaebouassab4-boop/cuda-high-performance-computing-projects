#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int N = 1024;  // Change this to benchmark different sizes (e.g., 512, 2048)

    printf("Matrix size N = %d\n", N);

    size_t size = N * (size_t)N * sizeof(float);
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize A and B with 1.0f
    for (size_t i = 0; i < N * (size_t)N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    clock_t start = clock();

    // Naive matrix multiplication
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    // Expected value: every element in C should be exactly N (since all 1s)
    printf("Expected C[i][j] = %.0f\n", (float)N);
    printf("CPU Execution Time: %.6f seconds\n", time_spent);

    // Optional: print a few elements to verify (top-left and bottom-right corners)
    printf("\nVerification (sample elements):\n");
    printf("C[0][0] = %.1f\n", C[0]);
    printf("C[0][%d] = %.1f\n", N-1, C[N-1]);
    printf("C[%d][0] = %.1f\n", N-1, C[(N-1)*N]);
    printf("C[%d][%d] = %.1f\n", N-1, N-1, C[N*N - 1]);

    free(A);
    free(B);
    free(C);
    return 0;
}
