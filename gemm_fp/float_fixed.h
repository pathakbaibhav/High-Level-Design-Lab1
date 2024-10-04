#pragma once

#include <cstdint>

// Roundup function
int roundup(float fp_number);

// Convert from float to fixed
void mm_float_to_fixed(const float* A_float, int32_t* A_fixed, int size, int scale);

// Convert from fixed to float
void mm_fixed_to_float(const int32_t* A_fixed, float* A_float, int size, int scale);