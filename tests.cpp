#include "matmul.h"
#include <iostream>
#include <cstdint>
using namespace std;


using MatrixFn = void(*)(const float*, const float*, float*, size_t, size_t, size_t);


void test_matrix8x8(MatrixFn fn){
    float A[24] = {
        1, 2, 3, 
        4, 5, 6, 
        7, 8, 9, 
        10, 11, 12, 
        13, 14, 15, 
        16, 17, 18, 
        19, 20, 21, 
        22, 23, 24
    };
    float B[24] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24
    };

    float result_output[64] = {0};

    float expected_out[64] = {
    70, 76, 82, 88, 94, 100, 106, 112,
    151, 166, 181, 196, 211, 226, 241, 256,
    232, 256, 280, 304, 328, 352, 376, 400,
    313, 346, 379, 412, 445, 478, 511, 544,
    394, 436, 478, 520, 562, 604, 646, 688,
    475, 526, 577, 628, 679, 730, 781, 832,
    556, 616, 676, 736, 796, 856, 916, 976,
    637, 706, 775, 844, 913, 982, 1051, 1120  
    };


    fn(A, B, result_output, 8, 3, 8);
    cout << "Result matrix(8x8):" << endl;
    print_matrix(result_output, 8, 8);
    cout << endl << "Expected matrix(8x8):" << endl;
    print_matrix(expected_out, 8, 8);
}


void test_rand(MatrixFn fn, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols) {
    float* A = rand_matrix(A_Rows, A_Cols_B_Rows);
    float* B = rand_matrix(A_Cols_B_Rows, B_Cols);
    float* C = new float[A_Rows * B_Cols];
    float* D = new float[A_Rows * B_Cols];
    fn(A, B, C, A_Rows, A_Cols_B_Rows, B_Cols);
    naive_matmul(A, B, D, A_Rows, A_Cols_B_Rows, B_Cols);

    cout << "A: \n";
    print_matrix(A, A_Rows, A_Cols_B_Rows);
    cout << "B: \n";
    print_matrix(B, A_Cols_B_Rows, B_Cols);
    cout << "A * B: \n";
    print_matrix(C, A_Rows, B_Cols);

    //compare results to naive since I know its working.
    cout << endl <<"Expected output: \n";
    print_matrix(D, A_Rows, B_Cols);

    delete[] C;
    delete[] D;
}


int main() {


    //test_matrix8x8(simd_matmul); // Change to differnt matrix algorithms for testing.
    test_rand(simd_matmul, 2, 8, 8);

    return 0;
}