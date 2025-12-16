#include <stdio.h>

int main() {

    // Declare 2x2 matrices A, B, and C
    float A[2][2], B[2][2], C[2][2];
    int i, j, k;

    // Input matrix A
    printf("Enter elements of 2x2 matrix A (row-wise):\n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            scanf("%f", &A[i][j]);
        }
    }

    // Input matrix B
    printf("Enter elements of 2x2 matrix B (row-wise):\n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            scanf("%f", &B[i][j]);
        }
    }

    // Matrix multiplication: C = A × B
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            C[i][j] = 0;  // Initialize result cell
            for (k = 0; k < 2; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Print the resulting matrix C
    printf("Resultant 2x2 matrix C:\n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            printf("%.2f ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
