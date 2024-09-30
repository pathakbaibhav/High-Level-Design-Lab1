#include "float_fixed.h"
#include <math.h>

int roundup(float fp_number)
{
	int	fx_number	=	(int)fp_number;

	if(fp_number-fx_number>=0.5)	fx_number++;

	return	fx_number;
}

void mm_float_to_fixed (float* A_float, int* A_fixed, int M, int K, int scale) {
	int length = M*K;

	for (int i=0; i<length; i++) {
		A_fixed[i] = roundup(A_float[i]*(1<<scale));
	}
}


/** Fixed to floating point */
void mm_fixed_to_float (int* A_fixed, float* A_float, int M, int K, int scale) {
	int length = M*K;

	for (int i=0; i<length; i++) {
		A_float[i] = (float)A_fixed[i]/(1<<scale);
	}
}