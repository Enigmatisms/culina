/**
 * Mat * Vector -> SGEMV, the purpose is to generate
 * several optimized functions for different workloads (M, N)
 * for different N length. For example:
 * for great N (1024+), we might need to break one row into several blocks
 * for median N (128 - 1024), one block can process a single row (multiply and warp reduce)
 * for small N (< 128), one block might need to process more than one row
*/

#include "utils.h"
#include <random>
#include <iostream>
#include <cuda_runtime.h>

#define SGEMV_TEST 1
#define OMP_THREADS 16
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__device__ __forceinline__ float warp_reduce(float value, int start_mask = 16) {
    #pragma unroll
    for (int mask = start_mask; mask >= 1; mask >>= 1) {
        int float_int = __float_as_int(value);
        float_int = __shfl_xor_sync(0xffffffff, float_int, mask);
        value += __int_as_float(float_int);
    }
    return value;
}

// 1024 needs to be subdivided into 4 blocks
// one way, we can subdivide the input by blocks
// the other way is to use thread coarsening
template <int threadNum>
__global__ void sgemv_kernel_long(float* A, float* p, float* v, int cols) {
    const int gmem_v_base = blockIdx.x * threadNum, gmem_a_base = blockIdx.y * cols + gmem_v_base, reduce_len = threadNum >> 5;
    __shared__ float reduced[reduce_len];
    // multiply
    float value = A[gmem_a_base + threadIdx.x] * p[gmem_v_base + threadIdx.x];
    __syncthreads();
    // warp reduce
    value = warp_reduce(value);
    const int warp_id = threadIdx.x >> 5;
    if (threadIdx.x % 32 == 0)
        reduced[warp_id] = value;
    __syncthreads();
    if (!warp_id) {
        float input_value = threadIdx.x < reduce_len ? reduced[threadIdx.x] : 0.f;
        value = warp_reduce(input_value, reduce_len >> 1);
        if (threadIdx.x == 0)
            atomicAdd(v + blockIdx.y, value);
    }
} 

/**
 * This implementation is based on thread coarsening
 * Each thread will take care of 4 consecutive positions
 * and one block process one single thread (1024 cols, tops)
*/
template <int threadNum = 256>
__global__ void sgemv_kernel_long_vec4(float* A, float* p, float* v, int cols) {
    constexpr int reduce_len = threadNum >> 5;
    const int gmem_base = blockIdx.x * cols;
    __shared__ float reduced[reduce_len];
    // multiply
    float4 A_part = FLOAT4(A[gmem_base + (threadIdx.x << 2)]);
    float4 p_part = FLOAT4(p[threadIdx.x << 2]);
    float value = A_part.x * p_part.x + A_part.y * p_part.y + A_part.z * p_part.z + A_part.w * p_part.w;
    __syncthreads();
    // warp reduce
    value = warp_reduce(value);
    const int warp_id = threadIdx.x >> 5;
    if (threadIdx.x % 32 == 0)          // lane is 0
        reduced[warp_id] = value;
    __syncthreads();
    if (!warp_id) {
        float input_value = threadIdx.x < reduce_len ? reduced[threadIdx.x] : 0.f;
        value = warp_reduce(input_value, reduce_len >> 1);
        if (threadIdx.x == 0)
            v[blockIdx.x] = value;
    }
    // ======== actually, since the warp reduced part (second) is too small, there is almost no speed difference =====
    // if (threadIdx.x == 0) {
    //     float result = 0;
    //     for (int i = 0; i < 8; i++) {
    //         result += reduced[i];
    //     }
    //     v[blockIdx.x] = value;
    // }
} 

__global__ void sgemv_kernel_naive(float* A, float* p, float* v, int cols) {
    int row_id = threadIdx.x << 2;
    for (int i = 0; i < 4; i++) {
        int row = row_id + i;
        float result = 0;
        for (int j = 0; j < cols; j++) {
            result += A[row * cols + j] * p[j];
        }
        v[row] = result;
    }
} 

// single warp reduce (N = 32), one block can process several lines
template <int threadNum>
__global__ void sgemv_kernel_warp(float* A, float* p, float* v) {
    const int gmem_addr_a = blockIdx.x * threadNum + threadIdx.x, lane = threadIdx.x % 32, 
              warp_id = threadIdx.x >> 5, num_rows = threadNum >> 5;
    // multiply and store locally
    float value = A[gmem_addr_a] * p[lane];
    __syncthreads();
    value = warp_reduce(value);

    if (lane == 0) {
        v[blockIdx.x * num_rows + warp_id] = value;
    }
} 

// half_warp reduce (N = 16): one warp process two rows
template <int threadNum>
__global__ void sgemv_kernel_half_warp(float* A, float* p, float* v) {
    const int gmem_addr_a = blockIdx.x * threadNum + threadIdx.x, lane = threadIdx.x % 16, 
              row_id = threadIdx.x >> 4, num_rows = threadNum >> 4;
    float value = A[gmem_addr_a] * p[lane];
    __syncthreads();
    value = warp_reduce(value, 8);

    if (lane == 0) {
        v[blockIdx.x * num_rows + row_id] = value;
    }
} 

float sgemv_host_caller(float* __restrict__ A, float* __restrict__ p, float* __restrict__ v, int M = 2048, int N = 1024) {
	float *devA, *devB, *devC;
	CUDA_CHECK_RETURN(cudaMalloc(&devA, sizeof(float) * M * N));
	CUDA_CHECK_RETURN(cudaMalloc(&devB, sizeof(float) * N));
	CUDA_CHECK_RETURN(cudaMalloc(&devC, sizeof(float) * M));

	CUDA_CHECK_RETURN(cudaMemcpy(devA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devB, p, sizeof(float) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemset(devC, 0, sizeof(float) * M));

    TicToc timer;
    timer.tic();
    #if SGEMV_TEST == 0
	    // long 1024: 2D atomic
        constexpr int num_threads = 256;
	    sgemv_kernel_long<num_threads><<<dim3(N / num_threads, M), num_threads>>>(devA, devB, devC, N);
    #elif SGEMV_TEST == 1
        // long 1024: 1D thread coarsening
        constexpr int num_threads = 256;
	    sgemv_kernel_long_vec4<num_threads><<<M, num_threads>>>(devA, devB, devC, N);
    #elif SGEMV_TEST == 2
        // warp 32 (col): warp reduce and multi-line per block
        constexpr int num_threads = 256;
	    sgemv_kernel_warp<num_threads><<<M / (num_threads >> 5), num_threads>>>(devA, devB, devC);
    #elif SGEMV_TEST == 3
        // warp 16 (col): half warp reduce and multi-line per block
        constexpr int num_threads = 256;
	    sgemv_kernel_half_warp<num_threads><<<M / (num_threads >> 4), num_threads>>>(devA, devB, devC);
    #else
        constexpr int num_threads = 1024;
        sgemv_kernel_naive<<<1, num_threads>>>(devA, devB, devC, N);
    #endif // endif SGEMV_TEST macro
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    float elapsed = timer.toc();
	// implicit synchronize
	CUDA_CHECK_RETURN(cudaMemcpy(v, devC, sizeof(float) * M, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(devA));
	CUDA_CHECK_RETURN(cudaFree(devB));
	CUDA_CHECK_RETURN(cudaFree(devC));
    return elapsed;
}

void sgemv_cpu_multi_threading(
	float* A, float* p, float* v, 
	const int M, const int N
) {
	#pragma omp parallel for num_threads(OMP_THREADS)
    for (int m = 0; m < M; m++) {
        float psum = 0.0;
        for (int n = 0; n < N; n++) {
            psum += A[m * N + n] * p[n];
        }
        v[m] = psum;
    }
}

int main() {
    // todo

    omp_set_num_threads(OMP_THREADS);
	int M = 4096, N = 1024;

	float *A, *p, *v, *w;

    CUDA_CHECK_RETURN(cudaMallocHost(&A, sizeof(float) * M * N));
    CUDA_CHECK_RETURN(cudaMallocHost(&p, sizeof(float) * N));
    CUDA_CHECK_RETURN(cudaMallocHost(&v, sizeof(float) * M));
    CUDA_CHECK_RETURN(cudaMallocHost(&w, sizeof(float) * M));

	printf("Generating random matrix A...\n");
	generate_random_matrix(A, M, N);
	printf("Generating random vector p...\n");
	generate_random_matrix(p, 1, N);
	printf("Setting v and w vector to zero...\n");
	memset(v, 0, sizeof(float) * M);
	memset(w, 0, sizeof(float) * M);

    TicToc timer;

	printf("CPU SGEMV calculating.\n");
    for (int i = 0; i < 5; i++) {
        sgemv_cpu_multi_threading(A, p, v, M, N);
    }
	timer.tic();
    for (int i = 0; i < 10; i++) {
        sgemv_cpu_multi_threading(A, p, v, M, N);
    }
	float cpu_time_ms = timer.toc() / 10.0;
	printf("CPU SGEMV finished in %.5f ms.\n", cpu_time_ms);
	nan_checker(v, 1, M);

	printf("GPU SGEMV calculating.\n");
    for (int i = 0; i < 5; i++) {
        sgemv_host_caller(A, p, w, M, N);
    }
    float gpu_time_ms = 0;
    for (int i = 0; i < 10; i++) {
        gpu_time_ms += sgemv_host_caller(A, p, w, M, N);
    }
	printf("GPU SGEMV finished in %.5f ms.\n", gpu_time_ms / 10.f);
	nan_checker(w, 1, M);

	float diff = compare_result(v, w, M, 1);
	printf("GPU SGEMV: %f ms, CPU %d threads: %f ms. MAE: %.5f, AE: %.3f\n", gpu_time_ms, OMP_THREADS, cpu_time_ms, diff, diff * M);

    CUDA_CHECK_RETURN(cudaFreeHost(A));
    CUDA_CHECK_RETURN(cudaFreeHost(p));
    CUDA_CHECK_RETURN(cudaFreeHost(v));
    CUDA_CHECK_RETURN(cudaFreeHost(w));

    return 0;
}