#include "float_fixed.h"

int32_t roundup (float fp_number) 
{
    int32_t fx_number = (int32_t)(fp_number);
  
    if (fp_number - fx_number >= 0.5) fx_number++;

    return fx_number;
}

void mm_float_to_fixed (const float* A_float, int32_t* A_fixed, int M, int N, int scale)
{
    int size = M * N;
    for (int i = 0; i < size; i++)
    {
        A_fixed[i] = roundup(A_float[i] * (1 << scale));
    }
}

void mm_fixed_to_float (const int32_t* A_fixed, float* A_float, int M, int N, int scale)
{
    int size = M * N;
    for (int i = 0; i < size; i++)
    {
        A_float[i] = (float)A_fixed[i] / (1 << scale);
    }
}

