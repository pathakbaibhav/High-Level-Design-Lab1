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

	float FloatPointValue = 0.213;
	int scale = 16;
	
	// mapping
	int FixedPointValue = roundup(FloatPointValue*(1<<scale));
	
	// here is your fixed point algorithm
	FixedPointValue = FixedPointValue * 500;
	
	// remapping
	float result = (float)(FixedPointValue)/(1<<scale);
	
	// printing result
	printf("%.10f\n",FloatPointValue*500);
	printf("%.10f\n",result);

	float ALPHA=1.0;
	float BETA=0.0;

	int lda = tst_dim_K;
    int ldb = tst_dim_N;
    int ldc = tst_dim_N;

	// Call gemm_cpu
	cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, ALPHA, tst_matrix_A, lda, tst_matrix_B, ldb, BETA, tst_matrix_Cin, ldc);

	printf("SNR for float gemm is %.10f dB\n", matrix_snr(tst_matrix_Cin, tst_matrix_Cout, tst_dim_M, tst_dim_N));

	// int scale = 16;
	int bestScale=-1;
	float bestSNR= -10000.0;

	for(int scale=1; scale < 32; scale++){
		long int* A_fixed = (long int*)malloc(tst_dim_M * tst_dim_K * sizeof(long int));	// Fixed point arrays
	    long int* B_fixed = (long int*)malloc(tst_dim_K * tst_dim_N * sizeof(long int));
		long int* Cin_fixed = (long int*)calloc(tst_dim_M * tst_dim_N, sizeof(long int)); // Calloc to clean the memory

		mm_float_to_fixed(tst_matrix_A, A_fixed, tst_dim_M, tst_dim_K, scale);	// Populate fixed-point A and B
		mm_float_to_fixed(tst_matrix_B, B_fixed, tst_dim_K, tst_dim_N, scale);

		fixed_cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, ALPHA, A_fixed, lda, B_fixed, ldb, BETA, Cin_fixed, ldc);	// gemm fixed

		mm_fixed_to_float(Cin_fixed, tst_matrix_Cin, tst_dim_M, tst_dim_N, scale<<1);	// Convert output back to fixed, double scale

		float snr = matrix_snr(tst_matrix_Cin, tst_matrix_Cout, tst_dim_M, tst_dim_N);

		printf("SNR for fixed gemm is %.10f dB with scale=%d\n", snr, scale);	// SNR 

		// Record max values
		if (snr >= bestSNR) {
			bestScale = scale;
			bestSNR = snr;
		}

		free(A_fixed);
	    free(B_fixed);
	    free(Cin_fixed);
		}

	printf("Best snr is %.10f with scale=%d\n", bestSNR, bestScale);
}