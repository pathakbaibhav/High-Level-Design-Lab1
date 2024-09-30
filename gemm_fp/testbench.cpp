#include <stdio.h>
#include <math.h>

#include "gemm_cpu_fp.h"
#include "testData.h"

int roundup(float fp_number)
{
	int	fx_number	=	(int)fp_number;

	if(fp_number-fx_number>=0.5)	fx_number++;

	return	fx_number;
}

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
	// how to compute 0.213 * 500 in fixed point?
	/*
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

	float ALPHA=1.0;
	float BETA=0.0;

	int lda = tst_dim_K;
    int ldb = tst_dim_N;
    int ldc = tst_dim_N;

	// Call gemm_cpu
	cpu_gemm_nn(0, 0, tst_dim_M, tst_dim_N, tst_dim_K, ALPHA, tst_matrix_A, lda, tst_matrix_B, ldb, BETA, tst_matrix_Cin, ldc);

	// for(unsigned int i=0; i<tst_dim_N; i=i+40){
	// 	printf("gemm output: %.10f, actual: %.10f\n", tst_matrix_Cin[i], tst_matrix_Cout[i]);
	// }

	printf("SNR is %.10f dB\n", matrix_snr(tst_matrix_Cin, tst_matrix_Cout, tst_dim_M, tst_dim_N));


}