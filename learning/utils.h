#include <omp.h>
#include <chrono>
#include <random>
#include <iostream>

#ifndef NO_CUDA
__host__ static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__host__ static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	printf("%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err), err, file, line);
	exit (1);
}
#endif

class TicToc {
private:
    std::chrono::system_clock::time_point tp;
public:
    void tic() {
        tp = std::chrono::system_clock::now();
    }

    double toc() const {
        auto dur = std::chrono::system_clock::now() - tp;
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
        return static_cast<double>(count) / 1e3;
    }
};

float compare_result(
	float* A, float* B, const int M, const int N
) {
	float diff = 0;
	#pragma omp parallel for num_threads(16) reduction(:+diff)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
			int addr = m * N + n;
			diff += fabsf(A[addr] - B[addr]);
        }
    }
	return diff / (M * N);
}

void generate_random_matrix(float* mat, const int rows, const int cols) {
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 5);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			mat[i * cols + j] = dis(gen);
		}
	}
}

void nan_checker(float* mat, const int rows, const int cols) {
	int num_nan_inf = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			float val = mat[i * cols + j];
			num_nan_inf += std::isnan(val) || std::isinf(val);
		}	
	}
	printf("NaN ratio: %.3f%%\n", float(num_nan_inf) / float(rows * cols) * 100.f);
}
