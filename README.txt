There are two ways to run the code, either using the tests.cpp to run matrix multiplication on a random matrix with input and output displayed. This is used for testing/verifying the algorithms when I made them. Or by running timings.cpp which requires input arguments to run (matrix sizes). keep in mind, it has to be a multiple of 8. There is also a python script batch_timing.py that will run the timings.cpp over multiple matrix sizes.


Enabling CUDA:
By default the CUDA matrix multiplication is disabled. In order to run uncomment the //#include "cuda_matmul.h" and use the nvcc to compile.


How to run:
in the main function (either tests.cpp or timings.cpp) replace the function pointer with whichever algorithm you'd like to tests. Then compile.


How to compile:
	non-cuda version:
g++ -Ivectorclass -std=c++20 -Wall -Wextra -Wpedantic -O3 timings.cpp matmul.cpp -o matmul_bench

	cuda version:
nvcc -Ivectorclass -std=c++20 -O3 cuda_matmul.cu timings.cpp matmul.cpp -o matmul_bench