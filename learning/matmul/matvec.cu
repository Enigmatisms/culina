/**
 * Mat * Vector -> SGEMV, the purpose is to generate
 * several optimized functions for different workloads (M, N)
 * for different N length. For example:
 * for great N (1024+), we might need to break one row into several blocks
 * for median N (128 - 1024), one block can process a single row (multiply and warp reduce)
 * for small N (< 128), one block might need to process more than one row
*/

#include "../utils.h"
#include <random>
#include <iostream>
#include <cuda_runtime.h>

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__device__ float warp_reduce(float value, int start_mask = 16) {
    #pragma unroll
    for (int mask = start_mask; mask >= 1; mask >> 1) {
        int float_int = __float_as_int(value);
        int float_int = __shfl_xor_sync(0xffffffff, float_int, mask);
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
    float value = warp_reduce(value);
    const int warp_id = threadIdx.x >> 5;
    if (threadIdx.x % 32 == 0)
        reduced[warp_id] = value;
    __syncthreads();
    if (!warp_id) {
        float input_value = threadIdx.x < reduce_len ? reduced[threadIdx.x] : 0.f;
        float value = warp_reduce(input_value, reduce_len >> 1);
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
    const int gmem_base = blockIdx.x * cols, reduce_len = threadNum >> 5;
    __shared__ float reduced[reduce_len];
    // multiply
    float4 A_part = FLOAT4(A[gmem_base + (threadIdx.x << 2)]);
    float4 p_part = FLOAT4(p[threadIdx.x << 2]);
    float value = A_part.x * p_part.x + A_part.y * p_part.y + A_part.z * p_part.z + A_part.w * p_part.w;
    __syncthreads();
    // warp reduce
    float value = warp_reduce(value);
    const int warp_id = threadIdx.x >> 5;
    if (threadIdx.x % 32 == 0)
        reduced[warp_id] = value;
    __syncthreads();
    if (!warp_id) {
        float input_value = threadIdx.x < reduce_len ? reduced[threadIdx.x] : 0.f;
        value = warp_reduce(input_value, reduce_len >> 1);
        v[blockIdx.x] = value;
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

void sgemm_host_caller(float* __restrict__ A, float* __restrict__ p, float* __restrict__ v, int M = 2048, int N = 1024) {
	float *devA, *devB, *devC;
	CUDA_CHECK_RETURN(cudaMalloc(&devA, sizeof(float) * M * N));
	CUDA_CHECK_RETURN(cudaMalloc(&devB, sizeof(float) * N));
	CUDA_CHECK_RETURN(cudaMalloc(&devC, sizeof(float) * M));

	CUDA_CHECK_RETURN(cudaMemcpy(devA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devB, p, sizeof(float) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devC, v, sizeof(float) * M, cudaMemcpyHostToDevice));

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
    #else
        // warp 16 (col): half warp reduce and multi-line per block
        constexpr int num_threads = 256;
	    sgemv_kernel_half_warp<num_threads><<<M / (num_threads >> 4), num_threads>>>(devA, devB, devC);
    #endif // endif SGEMV_TEST macro

	// implicit synchronize
	CUDA_CHECK_RETURN(cudaMemcpy(v, devC, sizeof(float) * M, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(devA));
	CUDA_CHECK_RETURN(cudaFree(devB));
	CUDA_CHECK_RETURN(cudaFree(devC));
}

int main() {
    // todo
    #define SGEMV_TEST 0

    
    return 0;
}