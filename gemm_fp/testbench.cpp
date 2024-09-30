#include <stdio.h>
#include <math.h>

#include "gemm_cpu_fp.h"
#include "testData.h"
#include "float_fixed.h"

float matrix_snr(float* uut_gemm_cout, float* tst_matrix_Cout, int M, int N) {
	// ddif = uut_gemm_cout – tst_matrix_Cout;
	// disp([‘SNR is’, num2str(10*log10(sum(tst_matrix_Cout(:).^2)/sum(ddiff(:).^2))), ‘dB’]);

	float signal_power = 0.0;	// Signal strength 
    float noise_power = 0.0;	// Noise strength
    int length = M*N;

    for (int i = 0; i < length; i++) {
        signal_power += tst_matrix_Cout[i] * tst_matrix_Cout[i];      // Square signal
        float diff = uut_gemm_cout[i] - tst_matrix_Cout[i];           // Calculate noise
        noise_power += diff * diff;                                   // Square noise
    }

    // SNR in dB
    float snr = 10 * std::log10f(signal_power / noise_power);

    return snr;
}

int main(int argc, char* argv[])
{
	float ALPHA=1.0;
	float BETA=0.0;

	int lda = tst_dim_K;
    int ldb = tst_dim_N;
    int ldc = tst_dim_N;

	// Call gemm_cpu
	cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, ALPHA, tst_matrix_A, lda, tst_matrix_B, ldb, BETA, tst_matrix_Cin, ldc);

	printf("SNR for float gemm is %.10f dB\n", matrix_snr(tst_matrix_Cin, tst_matrix_Cout, tst_dim_M, tst_dim_N));

	int scale = 16;

	int* A_fixed = (int*)malloc(tst_dim_M * tst_dim_K * sizeof(int));	// Fixed point arrays
    int* B_fixed = (int*)malloc(tst_dim_K * tst_dim_N * sizeof(int));
    int* Cin_fixed = (int*)malloc(tst_dim_M * tst_dim_N * sizeof(int));

	mm_float_to_fixed(tst_matrix_A, A_fixed, tst_dim_M, tst_dim_K, scale);	// Populate fixed-point A and B
	mm_float_to_fixed(tst_matrix_B, B_fixed, tst_dim_K, tst_dim_N, scale);

	fixed_cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, ALPHA, A_fixed, lda, B_fixed, ldb, BETA, Cin_fixed, ldc);	// gemm fixed

	mm_fixed_to_float(Cin_fixed, tst_matrix_Cin, tst_dim_M, tst_dim_N, scale);	// Convert output back to fixed

	printf("SNR for fixed gemm is %.10f dB\n", matrix_snr(tst_matrix_Cin, tst_matrix_Cout, tst_dim_M, tst_dim_N));	// SNR 

	free(A_fixed);
    free(B_fixed);
    free(Cin_fixed);
}