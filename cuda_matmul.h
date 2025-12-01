#ifndef CUDA_MATMUL_H
#define CUDA_MATMUL_H

#include <cstddef>
#include "vectorclass.h"


void cuda_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols);


#endif
