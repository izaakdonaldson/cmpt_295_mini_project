#include <cuda_runtime.h>
#include <stdio.h>


typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


#define BLOCK_SIZE 16


__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C);


__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    float Cvalue = 0.0f;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= A.height || col >= B.width)
        return;

    for (int e = 0; e < A.width; ++e) {
        Cvalue += A.elements[row * A.width + e] *
                  B.elements[e * B.width + col];
    }

    C.elements[row * C.width + col] = Cvalue;
}


void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;

    size_t sizeA = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, sizeA);
    cudaMemcpy(d_A.elements, A.elements, sizeA, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;

    size_t sizeB = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, sizeB);
    cudaMemcpy(d_B.elements, B.elements, sizeB, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;

    size_t sizeC = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, sizeC);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(B.width  / BLOCK_SIZE,
              A.height / BLOCK_SIZE);

    MatMulKernel<<<grid, block>>>(d_A, d_B, d_C);

    cudaMemcpy(C.elements, d_C.elements, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


__global__
void cuda_matmul_kernel(const float* A, const float* B, float* C,
                        size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= A_Rows || col >= B_Cols)
        return;

    float sum = 0.0f;
    for (size_t k = 0; k < A_Cols_B_Rows; ++k) {
        sum += A[row * A_Cols_B_Rows + k] *
               B[k   * B_Cols           + col];
    }

    C[row * B_Cols + col] = sum;
}


void cuda_matmul(const float* A, const float* B, float* Output, size_t A_Rows, size_t A_Cols_B_Rows, size_t B_Cols) {
    float *dA, *dB, *dC;

    size_t sizeA = A_Rows * A_Cols_B_Rows * sizeof(float);
    size_t sizeB = A_Cols_B_Rows * B_Cols * sizeof(float);
    size_t sizeC = A_Rows * B_Cols * sizeof(float);

    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((B_Cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (A_Rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cuda_matmul_kernel<<<grid, block>>>(dA, dB, dC,
                                        A_Rows, A_Cols_B_Rows, B_Cols);

    cudaMemcpy(Output, dC, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

