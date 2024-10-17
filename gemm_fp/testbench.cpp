#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cmath.h>

#include "gemm_cpu_fp.h"
#include "testData.h"
#include "float_fixed.h"

float matrix_snr(const float* uut_gemm_cout, const float* tst_matrix_Cout, int M, int N) 
{
	  float signal_power = 0.0;	// Signal strength 
    float noise_power = 0.0;	// Noise strength
    int length = M*N;

    for (int i = 0; i < length; i++) {
        signal_power += tst_matrix_Cout[i] * tst_matrix_Cout[i];      // Square signal
        float diff = uut_gemm_cout[i] - tst_matrix_Cout[i];           // Calculate noise
        noise_power += diff * diff;                                   // Square noise
    }

    if (noise_power == 0)	return INFINITY;						// To handle perfect match

    float snr = 10 * std::log10f(signal_power / noise_power); // SNR in dB
  
    return snr;
}

int main(int argc, char* argv[])
{	
  float ALPHA = 1.0;
  float BETA = 0.0;

	int lda = tst_dim_K;
	int ldb = tst_dim_N;
	int ldc = tst_dim_N;

	float snr;

	// Calling GEMM function
	cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, ALPHA, tst_matrix_A, lda, tst_matrix_B, ldb, BETA, tst_matrix_Cin, ldc);

  // SNR for floating
	snr = matrix_snr(tst_matrix_Cin, tst_matrix_Cout, tst_dim_M, tst_dim_N);
  printf("SNR for float gemm is %.10f dB\n", snr);
  
  // Find best scale for fixed
	float best_snr = -std::numeric_limits<float>::infinity();
	int best_scale = 0;

	for (int scale = 1; scale <= 32; scale++)
	{
		// Allocate memory for fixed-point matrices
		int32_t* A_fixed = (int32_t*)malloc(tst_dim_M * tst_dim_K * sizeof(int32_t));
		int32_t* B_fixed = (int32_t*)malloc(tst_dim_K * tst_dim_N * sizeof(int32_t));
		int32_t* Cin_fixed = (int32_t*)calloc(tst_dim_M * tst_dim_N, sizeof(int32_t)); // Calloc to clean the memory

		// Convert input matrices to fixed-point
		mm_float_to_fixed(tst_matrix_A, A_fixed, tst_dim_M, tst_dim_K, scale);
		mm_float_to_fixed(tst_matrix_B, B_fixed, tst_dim_K, tst_dim_N, scale);

		// Perform fixed-point GEMM
		fixed_cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, ALPHA, A_fixed, lda, B_fixed, ldb, BETA, Cin_fixed, ldc);

		// Convert result back to float for SNR calculation (using scale * 2)
		mm_fixed_to_float(Cin_fixed, tst_matrix_Cin, tst_dim_M, tst_dim_N, scale*2);

		// Calculate SNR
		snr = matrix_snr(tst_matrix_Cin, tst_matrix_Cout, tst_dim_M, tst_dim_N);
		printf("SNR for fixed gemm is %.10f dB with scale=%d\n", snr, scale);	// SNR 

		if (snr > best_snr) {
			best_snr = snr;
			best_scale = scale;
		}

		free(A_fixed);
    free(B_fixed);
    free(Cin_fixed);
	}

	printf("Best snr is %.10f with scale=%d\n", best_snr, best_scale);
}
