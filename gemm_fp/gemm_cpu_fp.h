#pragma once

void cpu_gemm_nn(int TA, int TB, int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc);