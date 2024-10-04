#ifndef GEMM_CPU_FP_H
#define GEMM_CPU_FP_H

void cpu_gemm_nn(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

/** Fixed point gemm */
void fixed_cpu_gemm_nn(int TA, int TB, int M, int N, int K, float ALPHA,
        long int *A, int lda,
        long int *B, int ldb,
        float BETA,
        long int *C, int ldc);

#endif