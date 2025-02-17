#include "gemm_cpu_fp.h"

#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif

void cpu_gemm_nn(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void fixed_cpu_gemm_nn(int TA, int TB, int M, int N, int K, int64_t ALPHA,
    int32_t *A, int lda,
    int32_t *B, int ldb,
    int64_t BETA,
    int32_t *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            PUT_IN_REGISTER int64_t A_PART = (ALPHA * A[i * lda + k]);
            for(j = 0; j < N; ++j)
            {
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}