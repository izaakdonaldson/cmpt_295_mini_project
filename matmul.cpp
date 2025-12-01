#include "matmul.h"
#include "vectorclass.h"
#include <iostream>
#include <cstddef>
#include <random>
using namespace std;

mt19937 rng(1234);
uniform_int_distribution<int> dist(0, 100);

// MATRIX IMPLIMENTIONS
// naive matrix
// simd matrix
// multithreaded matrix
// smid & multithreaded matrix
// naive block
// recursive block
// cuda matrix

__attribute__((optimize("no-tree-vectorize")))
void naive_matmul(const float* A, const float* B, float* Output,
                  size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols)
{
    float sum = 0;
    for (size_t i = 0; i < A_Rows; i++) {
        for (size_t j = 0; j < B_Cols; j++) {
            sum = 0.0f;
            for (size_t k = 0; k < A_Cols_B_Rows; k++) {
                sum += A[i*A_Cols_B_Rows+k] * B[k*B_Cols+j];
            }
            Output[i*B_Cols + j] = sum;
        }
    }
}

void simd_matmul(const float* A, const float* B, float* Output,
                 size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols)
{
    for (size_t i = 0; i < A_Rows; i++) {
        for (size_t j = 0; j < B_Cols; j += 8) {
            Vec8f acc(0.0f);
            for (size_t k = 0; k < A_Cols_B_Rows; k++) {
                float a = A[i*A_Cols_B_Rows + k];
                Vec8f b_vec = Vec8f().load(&B[k*B_Cols + j]);
                acc += b_vec * a;
            }
            acc.store(&Output[i*B_Cols + j]);
        }
    }
}




// ####################### utility functions #######################

// create random matrix
float* rand_matrix(size_t rows, size_t cols){
    float* mat = new float[rows * cols];
    for(size_t i = 0; i < rows * cols; i++){
        mat[i] = dist(rng);
    }
    return mat;
}

// print matrix
void print_matrix(const float* M, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            cout << M[r * cols + c] << " ";
        }
        cout << endl;
    }
}