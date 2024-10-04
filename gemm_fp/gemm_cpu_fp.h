#pragma once

#include "float_fixed.h"

void cpu_gemm_nn(int TA, int TB, int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc);

void fixed_cpu_gemm_nn(int TA, int TB, int M, int N, int K, int64_t ALPHA,
    int32_t *A, int lda,
    int32_t *B, int ldb,
    int64_t  BETA,
    int32_t *C, int ldc);
