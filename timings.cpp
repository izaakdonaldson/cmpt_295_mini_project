#define _POSIX_C_SOURCE 199309L
#include "matmul.h"
//#include "cuda_matmul.h"
#include <time.h>
#include <iostream>
#include <string>

using namespace std;

using MatrixFn = void(*)(const float*, const float*, float*, size_t, size_t, size_t);

void benchmark(MatrixFn fn, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols) {
    float* A = rand_matrix(A_Rows, A_Cols_B_Rows);
    float* B = rand_matrix(A_Cols_B_Rows, B_Cols);
    float* C = new float[A_Rows * B_Cols];
    fn(A, B, C, A_Rows, A_Cols_B_Rows, B_Cols);

    delete[] A;
    delete[] B;
    delete[] C;
}

void warmup(){
    size_t warmupA_Rows = 128;
    size_t warmupA_Cols_B_Rows = 128;
    size_t warmupB_Cols = 128;

    float* warmupA = rand_matrix(warmupA_Rows, warmupA_Cols_B_Rows);
    float* warmupB = rand_matrix(warmupA_Cols_B_Rows, warmupB_Cols);
    float* warmupC = new float[warmupA_Rows * warmupB_Cols];

    for (int i = 0; i < 4; i++) {
        block_matmul(warmupA, warmupB, warmupC, warmupA_Rows, warmupA_Cols_B_Rows, warmupB_Cols);
    }
    delete[] warmupC;
}



// run on multiples of 64 * 10^x

int main(int /*argc*/, char** argv) {
    struct timespec start, end;
    size_t N = stoul(argv[1]);

    warmup();

    // Benchmark
    clock_gettime(CLOCK_MONOTONIC, &start);
    benchmark(mt_simd_matmul, N, N, N);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;
    cout << N << "," << elapsed << endl;
    //cout << N << "x" << N << ", took: " << elapsed << " Seconds" << endl;

    return 0;
}