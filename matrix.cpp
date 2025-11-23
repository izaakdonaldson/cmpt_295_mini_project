#include <iostream>
#include <random>
#include "vectorclass.h"
#include <cstdint>
using namespace std;

//random setup
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
void naive_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols){
    // i = A column index
    // j = A row & B column index
    // k = B row index
    float sum = 0;
    for (uint32_t i = 0; i < A_Rows; i++){
        for (uint32_t j = 0; j < B_Cols; j++){
            sum = 0.0;
            for (uint32_t k = 0; k < A_Cols_B_Rows; k++){
                sum += A[i * A_Cols_B_Rows + k] * B[k * B_Cols + j];
            }
            Output[i * B_Cols + j] = sum;
        }
    }
}


// assume B_Cols is divisible by 8
void simd_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols){
    // i = A column index
    // j = A row & B column index
    // k = B row index
    for (size_t i = 0; i < A_Rows; i++) {
        for (size_t j = 0; j < B_Cols; j += 8) {
            Vec8f acc(0.0f);
            for (size_t k = 0; k < A_Cols_B_Rows; k++) {
                float a_scalar = A[i * A_Cols_B_Rows + k];
                const float* Bptr = &B[k * B_Cols + j];
                Vec8f b_vec = Vec8f().load(Bptr);
                acc += b_vec * a_scalar;
            }
            acc.store(&Output[i * B_Cols + j]);
        }
    }
}


void print_matrix(const float* M, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            cout << M[r * cols + c] << " ";
        }
        cout << endl;
    }
}


using MatrixFn = void(*)(const float*, const float*, float*, size_t, size_t, size_t);

void test_matrix(MatrixFn fn){
    float A[4] = {
        1, 2,
        3, 4
    };
    float B[8] = {
        5, 6,
        7, 8
    };
    float C[6] = {
        5, 6, 7,
        8, 9, 10
    };
    float D[6] = {
        5, 6, 
        7, 8, 
        9, 10
    };

    float twoxtwo[4] = {0};
    float twoxthree[6] = {0};
    float threextwo[6] = {0};

    float one[4] = {19, 22, 43, 50};
    float two[6] = {21, 24, 27, 47, 54, 61};
    float three[6] = {23, 34, 31, 46, 39, 58};
    
    fn(A, B, twoxtwo, 2, 2, 2);
    cout << "Result matrix(2x2):" << endl;
    print_matrix(twoxtwo, 2, 2);
    cout << "Expected matrix(2x2):" << endl;
    print_matrix(one, 2, 2);

    fn(A, C, twoxthree, 2, 2, 3);
    cout << "Result matrix(2x3):" << endl;
    print_matrix(twoxthree, 2, 3);
    cout << "Expected matrix(2x3):" << endl;
    print_matrix(two, 2, 3);

    fn(D, A, threextwo, 3, 2, 2);
    cout << "Result matrix(3x2):" << endl;
    print_matrix(threextwo, 3, 2);
    cout << "Expected matrix(3x2):" << endl;
    print_matrix(three, 3, 2);
}

void test_matrix8(MatrixFn fn){
    float A[4] = {
        1, 2,
        3, 4
    };
    float B[16] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16
    };

    float C[24] = {
        1, 2, 3, 
        4, 5, 6, 
        7, 8, 9, 
        10, 11, 12, 
        13, 14, 15, 
        16, 17, 18, 
        19, 20, 21, 
        22, 23, 24
    };
    float D[24] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24
    };

    float twoxeight[16] = {0};
    float eightxeight[64] = {0};

    float out1[16] = {
        19, 22, 25, 28, 31, 34, 37, 40,
        39, 46, 53, 60, 67, 74, 81, 88
    };
    float out2[64] = {
    70, 76, 82, 88, 94, 100, 106, 112,
    151, 166, 181, 196, 211, 226, 241, 256,
    232, 256, 280, 304, 328, 352, 376, 400,
    313, 346, 379, 412, 445, 478, 511, 544,
    394, 436, 478, 520, 562, 604, 646, 688,
    475, 526, 577, 628, 679, 730, 781, 832,
    556, 616, 676, 736, 796, 856, 916, 976,
    637, 706, 775, 844, 913, 982, 1051, 1120  
    };


    fn(A, B, twoxeight, 2, 2, 8);
    cout << "Result matrix(2x8):" << endl;
    print_matrix(twoxeight, 2, 8);
    cout << "Expected matrix(2x8):" << endl;
    print_matrix(out1, 2, 8);

    fn(C, D, eightxeight, 8, 3, 8);
    cout << "Result matrix(8x8):" << endl;
    print_matrix(eightxeight, 8, 8);
    cout << "Expected matrix(8x8):" << endl;
    print_matrix(out2, 8, 8);
}

float* rand_matrix(size_t rows, size_t cols){
    float* mat = new float[rows * cols];
    for(size_t i = 0; i < rows * cols; i++){
        mat[i] = dist(rng);
    }
    return mat;
}

void test_rand(MatrixFn fn, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols) {
    float* A = rand_matrix(A_Rows, A_Cols_B_Rows);
    float* B = rand_matrix(A_Cols_B_Rows, B_Cols);
    float* C = new float[A_Rows * B_Cols];
    fn(A, B, C, A_Rows, A_Cols_B_Rows, B_Cols);
    cout << "A: \n";
    print_matrix(A, A_Rows, A_Cols_B_Rows);
    cout << "B: \n";
    print_matrix(B, A_Cols_B_Rows, B_Cols);
    cout << "A * B: \n";
    print_matrix(C, A_Rows, B_Cols);
}





int main() {

    //float* random = rand_matrix(2, 2);
    //print_matrix(random, 2, 2);
    //test_matrix8(simd_matmul); // Change to differnt matrix algorithms for testing.

    test_rand(naive_matmul, 2, 2, 2);

    return 0;
}