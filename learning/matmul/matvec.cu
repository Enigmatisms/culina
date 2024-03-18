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

__device__ float warp_reduce(float value) {
    for (int mask = 16; mask >= 1; mask >> 1) {
        int float_int = __float_as_int(value);
        int float_int = __shfl_xor_sync(0xffffffff, float_int, mask);
        value += __int_as_float(float_int);
    }
    return value;
}

// 1024 needs to be subdivided into 4 blocks
template <int threadNum>
__global__ void sgemv_kernel_1024(float* A, float* p, float* v, int M, int N) {
    const int gmem_v_base = blockIdx.x * threadNum, gmem_a_base = blockIdx.y * N + gmem_v_base, reduce_len = threadNum >> 5;
    __shared__ float reduced[reduce_len];
    // multiply
    float value = A[gmem_a_base + threadIdx.x] * p[gmem_v_base + threadIdx.x];
    __syncthreads();
    // warp reduce
    float value = warp_reduce(value);
    const int warp_id = threadIdx.x >> 5;
    if (threadIdx.x % 32 == 0)
        reduced[warp_id5] = value;
    __syncthreads();
    if (!warp_id) {
        float input_value = threadIdx.x < reduce_len ? reduced[threadIdx.x] : 0.f;
        float value = warp_reduce(input_value);
        if (threadIdx.x == 0)
            atomicAdd(v + blockIdx.y, value);
    }
} 

// single warp reduce
__global__ void sgemv_kernel_32() {
    
} 

// half_warp reduce
__global__ void sgemv_kernel_16() {
    
} 

int main() {
    // todo

    return 0;
}