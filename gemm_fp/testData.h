
/**
 * @file testData.h
 * @brief Test dataset for module test gemm_fp. Converted from A. Gerstlauer testbench.m. 
 * @version 0.1
 * @date 2022-09-16
 * 
 * 
 */

// Matrix dimensions. 
// Matrix A = M * K
// Matrix B = K * N
// Matrix C = M * N

extern unsigned int tst_dim_M;
extern unsigned int tst_dim_N;
extern unsigned int tst_dim_K;

// Each matrix is stored as a single dimensional array 
// tst_matrix_Cout = tst_matrix_A * tst_matrix_B + tst_matrix_Cin
// compare your output against tst_matrix_Cout

extern float tst_matrix_A[]; 
extern float tst_matrix_B[]; 
extern float tst_matrix_Cin[]; 
extern float tst_matrix_Cout[]; 