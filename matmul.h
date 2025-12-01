#ifndef MATMUL_H
#define MATMUL_H

#include <cstddef>
#include "vectorclass.h"

using MatrixFn = void(*)(const float*, const float*, float*, size_t, size_t, size_t);

void naive_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols);

void simd_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols);

void mt_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols);

void mt_simd_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols);

// block_matmul adapted from  Gagarine Yaikhom yaikhom.com
void block_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols);

// co_block_matmul adapted from MIT Steven G. Johnson
void co_block_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols);



// ####################### utility functions #######################

float* rand_matrix(size_t rows, size_t cols);

void print_matrix(const float* M, size_t rows, size_t cols);

#endif
