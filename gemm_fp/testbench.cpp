#include <stdio.h>
#include <math.h>

// Adding the header files created
#include <iostream>
#include "gemm_cpu_fp.h"
#include "testData.h"
#include "float_fixed.h"

using namespace std;

float calculate_snr(const float* uut_gemm_cout, const float* tst_matrix_Cout, int size)
{
	float signal_power = 0.0;	// Signal Strength
	float noise_power = 0.0;	// Noise Strength
	float diff;

	for (int i = 0; i < size; ++i)
	{
		signal_power += tst_matrix_Cout[i] * tst_matrix_Cout[i];	// Square Signal
		diff = uut_gemm_cout[i] - tst_matrix_Cout[i];				// Calculate Noise
		noise_power += diff * diff;									// Square Noise
	}

	if (noise_power == 0)	return INFINITY;						// To handle perfect match

	return 10 * log10f(signal_power / noise_power);
}

int main(int argc, char* argv[])
{
	/*
	// how to compute 0.213 * 500 in fixed point?
	
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
	*/

	/*
	// Assigning values for GEMM function call
	float ALPHA = 1.0, BETA = 1.0;
	*/

	int lda = tst_dim_K;
	int ldb = tst_dim_N;
	int ldc = tst_dim_N;

	float snr;

	// Calling GEMM function
	cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, 1.0, tst_matrix_A, lda, tst_matrix_B, ldb, 1.0, tst_matrix_Cin, ldc);

	/*
	// Print Sampled Comparison
	unsigned int i;
	cout << "Sampled Comparison: \n";
	for (i = 0; i < tst_dim_N; i += 25)
	{
		cout << "Function Call Output : " << tst_matrix_Cin[i] << "		Actual Output" << tst_matrix_Cout[i] << endl;
	}
	*/

	snr = calculate_snr(tst_matrix_Cin, tst_matrix_Cout, tst_dim_M*tst_dim_N);
	cout << "SNR for Float GEMM is: " << snr << " dB" << endl;

	int size = tst_dim_M * tst_dim_N;
	int size_A = tst_dim_M * tst_dim_K;
	int size_B = tst_dim_K * tst_dim_N;

	float best_snr = -std::numeric_limits<float>::infinity();
	int best_scale = 0;

	for (int scale = 1; scale <= 30; ++scale)
	{
		// Allocate memory for fixed-point matrices
		int32_t* A_fixed = new int32_t[size_A];
		int32_t* B_fixed = new int32_t[size_B];
		int32_t* C_fixed = new int32_t[size];
		float* C_result = new float[size];

		// Convert input matrices to fixed-point
		mm_float_to_fixed(tst_matrix_A, A_fixed, size_A, scale);
		mm_float_to_fixed(tst_matrix_B, B_fixed, size_B, scale);
		mm_float_to_fixed(tst_matrix_Cin, C_fixed, size, scale);

		// Perform fixed-point GEMM
		fixed_cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, 1.0, A_fixed, lda, B_fixed, ldb, 1.0, C_fixed, ldc, scale);

		// Convert result back to float for SNR calculation (using scale * 2)
		mm_fixed_to_float(C_fixed, C_result, size, scale * 2);

		// Calculate SNR
		snr = calculate_snr(tst_matrix_Cout, C_result, size);
		cout << "Scale: " << scale << "	SNR for Fixed GEMM is " << snr << " dB" << endl;

		if (snr > best_snr) {
			best_snr = snr;
			best_scale = scale;
		}

		delete[] A_fixed;
		delete[] B_fixed;
		delete[] C_fixed;
		delete[] C_result;
	}

	cout << "\nBest SNR: " << best_snr << " dB at scale " << best_scale << endl;
}