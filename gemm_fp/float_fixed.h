#pragma once

#include <cstdint>

/** Roundup from provided testbench.cpp file */
int32_t roundup(float fp_number);

/** Floating to fixed point */
void mm_float_to_fixed(const float* A_float, int32_t* A_fixed, int M, int N, int scale);

/** Fixed to floating point */
void mm_fixed_to_float(const int32_t* A_fixed, float* A_float, int M, int N, int scale);