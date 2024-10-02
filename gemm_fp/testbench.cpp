#include <stdio.h>
#include <math.h>

// Adding the header files created
#include <iostream>
#include "gemm_cpu_fp.h"
#include "testData.h"

using namespace std;

int roundup(float fp_number)
{
	int	fx_number =	(int)fp_number;
	if(fp_number-fx_number >= 0.5)	fx_number++;

	return	fx_number;
}

float matrix_snr(const float* uut_gemm_cout, const float* tst_matrix_Cout, int size)
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

	return 10 * log10(signal_power / noise_power);
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

	// Assigning values for GEMM funfction call
	float ALPHA = 1.0, BETA = 1.0;

	int lda = tst_dim_K;
	int ldb = tst_dim_N;
	int ldc = tst_dim_N;

	float snr;

	//Calling GEMM function
	cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, ALPHA, tst_matrix_A, lda, tst_matrix_B, ldb, BETA, tst_matrix_Cin, ldc);

	/*
	//Print Sampled Comparision
	unsigned int i;
	cout << "Sampled Comparision: \n";
	for (i = 0; i < tst_dim_N; i += 25)
	{
		cout << "Function Call Output : " << tst_matrix_Cin[i] << "		Actual Output" << tst_matrix_Cout[i] << endl;
	}
	*/

	snr = matrix_snr(tst_matrix_Cin, tst_matrix_Cout, tst_dim_M*tst_dim_N);
	cout << "SNR is " << snr << " db" << endl;
}