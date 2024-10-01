#ifndef FLOAT_FIXED_H
#define FLOAT_FIXED_H

/** Roundup from provided testbench.cpp file */
int roundup(float fp_number);

/** Floating to fixed point */
void mm_float_to_fixed (float* A_float, long long int* A_fixed, int M, int K, int scale=16);

/** Fixed to floating point */
void mm_fixed_to_float (long long int* A_fixed, float* A_float, int M, int K, int scale=16);

#endif