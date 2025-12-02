#include "matmul.h"
#include "vectorclass.h"
#include <iostream>
#include <cstddef>
#include <random>
#include <thread>
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


// naive matrix
__attribute__((optimize("no-tree-vectorize")))
void naive_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols) {
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


// simd matrix
void simd_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols) {
    for (size_t i = 0; i < A_Rows; i++) {
        for (size_t j = 0; j < B_Cols; j += 8) {
            Vec8f acc(0.0f);
            for (size_t k = 0; k < A_Cols_B_Rows; k++) {
                float a_element = A[i*A_Cols_B_Rows + k];
                Vec8f b_vector = Vec8f().load(&B[k*B_Cols + j]);
                acc += b_vector * a_element;
            }
            acc.store(&Output[i*B_Cols + j]);
        }
    }
}


// thread function for mt_matmul
void mt_thread(size_t start_index, size_t end_index, const float* A, const float* B, float* Output, size_t A_Cols_B_Rows, size_t B_Cols){
    float sum = 0.0f;
    for (size_t i = start_index; i < end_index; i++) {
        for (size_t j = 0; j < B_Cols; j++) {
            sum = 0.0f;
            for (size_t k = 0; k < A_Cols_B_Rows; k++) {
                sum += A[i*A_Cols_B_Rows + k] * B[k*B_Cols + j];
            }
            Output[i*B_Cols + j] = sum;
        }
    }
}


// multithreaded matrix
void mt_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols){
    const size_t num_threads = 8;
    thread threads[num_threads];

    size_t rows_per_thread = A_Rows / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start_index = t * rows_per_thread;
        size_t end_index   = start_index + rows_per_thread;
        threads[t] = thread(mt_thread, start_index, end_index, A, B, Output, A_Cols_B_Rows, B_Cols);
    }

    for (size_t t = 0; t < num_threads; t++) {
        threads[t].join();
    }
}


// thread function for mt_simd_matmul
void mt_simd_thread(size_t start_index, size_t end_index, const float* A, const float* B, float* Output, size_t A_Cols_B_Rows, size_t B_Cols){
    for (size_t i = start_index; i < end_index; i++) {
        for (size_t j = 0; j < B_Cols; j += 8) {
            Vec8f acc(0.0f);
            for (size_t k = 0; k < A_Cols_B_Rows; k++) {
                float a_element = A[i*A_Cols_B_Rows + k];
                const float* Bptr = &B[k*B_Cols + j];
                Vec8f b_vector = Vec8f().load(Bptr);
                acc += b_vector * a_element;
            }
            acc.store(&Output[i * B_Cols + j]);
        }
    }
}

// multithreaded and simd matrix
void mt_simd_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols){
    const size_t num_threads = 8;
    thread threads[num_threads];

    size_t rows_per_thread = A_Rows / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start_index = t * rows_per_thread;
        size_t end_index   = start_index + rows_per_thread;
        threads[t] = thread(mt_simd_thread, start_index, end_index, A, B, Output, A_Cols_B_Rows, B_Cols);
    }

    for (size_t t = 0; t < num_threads; t++) {
        threads[t].join();
    }
}


// helper function for cache oblivious block algorithm
void co_block_matmul_helper(const float* A, const float* B, float* C, size_t m, size_t n, size_t p, size_t fdA, size_t fdB, size_t fdC)
{
    // base case 16x16 matrix
    if (m + n + p <= 48) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t k = 0; k < p; ++k) {
                float acc = 0;
                for (size_t j = 0; j < n; ++j) {
                    acc += A[i*fdA + j] * B[j*fdB + k];
                }
                C[i*fdC + k] += acc;
            }
        }
        return;
    }

    else {
    // divide and conquer
    size_t m2 = m / 2;
    size_t n2 = n / 2;
    size_t p2 = p / 2;

    co_block_matmul_helper(A, B, C, m2, n2, p2, fdA, fdB, fdC);
    co_block_matmul_helper(A + n2, B + n2*fdB, C, m2, n - n2, p2, fdA, fdB, fdC);

    co_block_matmul_helper(A, B + p2, C + p2, m2, n2, p - p2, fdA, fdB, fdC);
    co_block_matmul_helper(A + n2, B + p2 + n2*fdB, C + p2, m2, n - n2, p - p2, fdA, fdB, fdC);

    co_block_matmul_helper(A + m2*fdA, B, C + m2*fdC, m - m2, n2, p2, fdA, fdB, fdC);
    co_block_matmul_helper(A + m2*fdA + n2, B + n2*fdB, C + m2*fdC, m - m2, n - n2, p2, fdA, fdB, fdC);

    co_block_matmul_helper(A + m2*fdA, B + p2, C + m2*fdC + p2, m - m2, n2, p - p2, fdA, fdB, fdC);
    co_block_matmul_helper(A + m2*fdA + n2, B + p2 + n2*fdB, C + m2*fdC + p2, m - m2, n - n2, p - p2, fdA, fdB, fdC);
    }
}

// cache oblivious block algorithm
// co_block_matmul & the helper function is adapted from MIT Steven G. Johnson
void co_block_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols){
    co_block_matmul_helper(A, B, Output, A_Rows, A_Cols_B_Rows, B_Cols, A_Cols_B_Rows, B_Cols, B_Cols);
}


// block_matmul is adapted from  Gagarine Yaikhom yaikhom.com
void block_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols){
    // block sizes for Ryzen-7 9800X3D
    // L1 Cache 48kb       max block size = 48
    // L2 Cache 1mb        max block size = 192
    // L3 Cache 96mb       max block size = 1536
    // Out of Cache        min block size = 4096
    const size_t s = 64;
    for (size_t i = 0; i < A_Rows * B_Cols; ++i)
        Output[i] = 0.0f;
    for (size_t ii = 0; ii < A_Rows; ii += s) {
        for (size_t jj = 0; jj < B_Cols; jj += s) {
            for (size_t kk = 0; kk < A_Cols_B_Rows; kk += s) {
                size_t ie = min(ii + s, A_Rows);
                size_t je = min(jj + s, B_Cols);
                size_t ke = min(kk + s, A_Cols_B_Rows);
                for (size_t i = ii; i < ie; ++i) {
                    for (size_t j = jj; j < je; ++j) {
                        float sum = 0.0f;
                        for (size_t k = kk; k < ke; ++k) {
                            sum += A[i * A_Cols_B_Rows + k] * B[k * B_Cols + j];
                        }
                        Output[i * B_Cols + j] += sum;
                    }
                }
            }
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