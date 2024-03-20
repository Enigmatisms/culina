/**
 * Vector Vector dot product (mult + reduce)
 * long and vec4 (131072) -> (128 block, 256 threads, reduce 8 -> reduce 1)
*/

#include "../utils.h"
#include <random>
#include <iostream>
#include <cuda_runtime.h>

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

// thread coarsening (factor 4)
template <int threadNum = 256>
__global__ void dot_prod_kernel_long(float* a, float* b, float* res) {
    constexpr int reduce_len = threadNum >> 5;
    const int gmem_addr = blockIdx.x * (threadNum << 2) + (threadIdx.x << 2);
    __shared__ float reduced[reduce_len];
    // multiply
    float4 a_part = FLOAT4(a[gmem_addr]);
    float4 b_part = FLOAT4(b[gmem_addr]);
    float value = a_part.x * b_part.x + a_part.y * b_part.y + a_part.z * b_part.z + a_part.w * b_part.w;
    __syncthreads();
    // warp reduce
    value = warp_reduce(value);
    const int warp_id = threadIdx.x >> 5;
    if (threadIdx.x % 32 == 0)
        reduced[warp_id] = value;
    __syncthreads();
    if (threadIdx.x == 0) {
        float result = 0;
        for (int i = 0; i < (threadNum >> 5); i++)
            result += reduced[i];
        atomicAdd(res, result);
    }
} 

float dot_prod_host_caller(float* __restrict__ a, float* __restrict__ b, float& result, int length = 131072) {
	float *dev_a, *dev_b, *dev_res;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_a, sizeof(float) * length));
	CUDA_CHECK_RETURN(cudaMalloc(&dev_b, sizeof(float) * length));
	CUDA_CHECK_RETURN(cudaMallocManaged(&dev_res, sizeof(float)));

	CUDA_CHECK_RETURN(cudaMemcpy(dev_a, a, sizeof(float) * length, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_b, b, sizeof(float) * length, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemset(dev_res, 0, sizeof(float)));

    TicToc timer;
    timer.tic();
    constexpr int num_threads = 128;
    dot_prod_kernel_long<num_threads><<<length / (num_threads << 2), num_threads>>>(dev_a, dev_b, dev_res);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    float elapsed = timer.toc();

    result = *dev_res;
	CUDA_CHECK_RETURN(cudaFree(dev_a));
	CUDA_CHECK_RETURN(cudaFree(dev_b));
	CUDA_CHECK_RETURN(cudaFree(dev_res));
    return elapsed;
}

float dot_prod_cpu_multi_threading(
	float* a, float* b, const int N
) {
    float result = 0;
	#pragma omp parallel for num_threads(OMP_THREADS) reduction(+: result)
    for (int n = 0; n < N; n++) {
        result += a[n] * b[n];
    }
    return result;
}

int main() {
    // todo

    omp_set_num_threads(OMP_THREADS);
	int N = 2097152;

	float *a, *b;

    CUDA_CHECK_RETURN(cudaMallocHost(&a, sizeof(float) * N));
    CUDA_CHECK_RETURN(cudaMallocHost(&b, sizeof(float) * N));

	printf("Generating random vector a...\n");
	generate_random_matrix(a, 1, N);
	printf("Generating random vector b...\n");
	generate_random_matrix(b, 1, N);

    TicToc timer;

	printf("CPU dot product calculating.\n");
    for (int i = 0; i < 5; i++) {
        dot_prod_cpu_multi_threading(a, b, N);
    }
	timer.tic();
    float cpu_result = 0;
    for (int i = 0; i < 10; i++) {
        cpu_result = dot_prod_cpu_multi_threading(a, b, N);
    }
	float cpu_time_ms = timer.toc() / 10.0;
	printf("CPU dot product finished in %.5f ms.\n", cpu_time_ms);

	printf("GPU dot product calculating.\n");
    float gpu_result = 0;
    for (int i = 0; i < 5; i++) {
        dot_prod_host_caller(a, b, gpu_result, N);
    }
    float gpu_time_ms = 0;
    for (int i = 0; i < 10; i++) {
        gpu_time_ms += dot_prod_host_caller(a, b, gpu_result, N);
    }
	printf("GPU dot product finished in %.5f ms.\n", gpu_time_ms / 10.f);

	printf("GPU dot product: %f ms, CPU %d threads: %f ms. GPU = %.6f, CPU = %6f\n", gpu_time_ms, OMP_THREADS, cpu_time_ms, gpu_result, cpu_result);

    CUDA_CHECK_RETURN(cudaFreeHost(a));
    CUDA_CHECK_RETURN(cudaFreeHost(b));

    return 0;
}
