#include <stdio.h>
#include <math.h>

//Adding the header files created
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

	//Calling GEMM function
	cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, ALPHA, tst_matrix_A, lda, tst_matrix_B, ldb, BETA, tst_matrix_Cin, ldc);

	//Print Sampled Comparision
	unsigned int i;
	cout << "Sampled Comparision: \n";
	for (i = 0; i < tst_dim_N; i += 25)
	{
		cout << "Function Call Output : " << tst_matrix_Cin[i] << "		Actual Output" << tst_matrix_Cout[i] << endl;
	}
}