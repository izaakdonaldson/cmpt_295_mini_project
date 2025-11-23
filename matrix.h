#ifndef MATRIX_H
#define MATRIX_H

void fill_matrix(double* matrix, size_t n, size_t m, uint32_t seed);

void nested_matrix_multiply(const double* M1, const double* M1, double* Output, size_t n, size_t m, size_t p);
void simd_matrix_multiply(const double* M1, const double* M1, double* Output, size_t n, size_t m, size_t p);
//void block_matrix_multiply(const double* M1, const double* M1, double* Output, size_t n, size_t m, size_t p);



// testing
using MatrixFn = void(*)(const float*, const float*, float*, size_t, size_t, size_t);
void test_matrix(MatrixFn fn)
void test_matrix8(MatrixFn fn);




#endif