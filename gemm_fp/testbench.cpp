#include <stdio.h>
#include <math.h>

int roundup(float fp_number)
{
	int	fx_number	=	(int)fp_number;

	if(fp_number-fx_number>=0.5)	fx_number++;

	return	fx_number;
}

int main(int argc, char* argv[])
{
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
}