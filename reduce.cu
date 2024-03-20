/**
 * CUDA reduce operation, I only implement int and float here
*/

#include <iostream>
#include <numeric>
#include <random>
#include "utils.h"

int next_multiple_of_32(int num) {
    int remainder = num % 32;
    if (remainder == 0)
        return num;
    return num + (32 - remainder);
}

__device__ __forceinline__ int warp_reduce_sum(int cur_thread_v, int warp_size = 32) {
    // warp reduce sum
    for (int mask = warp_size >> 1; mask >= 1; mask >>= 1) {
        cur_thread_v += __shfl_xor_sync(0xffffffff, cur_thread_v, mask, warp_size);
    }
    return cur_thread_v;
}

__device__ __forceinline__ int warp_reduce_max(int cur_thread_v, int warp_size = 32) {
    // warp reduce max
    for (int mask = warp_size >> 1; mask >= 1; mask >>= 1) {
        cur_thread_v = max(__shfl_xor_sync(0xffffffff, cur_thread_v, mask, warp_size), cur_thread_v);
    }
    return cur_thread_v;
}

__device__ __forceinline__ float warp_reduce_max(float cur_thread_v, int warp_size = 32) {
    // warp reduce max: via CUDA intrinsics
    for (int mask = warp_size >> 1; mask >= 1; mask >>= 1) {
        int float_int = __float_as_int(cur_thread_v);
        int recv_float = __shfl_xor_sync(0xffffffff, float_int, mask, warp_size);
        float res = __int_as_float(recv_float);
        cur_thread_v = fmaxf(res, cur_thread_v);
    }
    return cur_thread_v;
}

__device__ __forceinline__ float warp_reduce_sum(float value, int start_mask = 32) {
    #pragma unroll
    for (int mask = start_mask >> 1; mask >= 1; mask >>= 1) {
        int float_int = __float_as_int(value);
        float_int = __shfl_xor_sync(0xffffffff, float_int, mask);
        value += __int_as_float(float_int);
    }
    return value;
}

/**
 * TODO: modify the code and make it a block warp all reduce function (easy)
 * address for the vector will be offset by blockIdx.x * blockDim.x
*/
template <typename Ty, int WarpSize = 32>
__global__ void reduce_sum(const Ty* const vector, Ty* output, int length) {
    /**
     * length might be subdivided into several warps, and each warp will first have reduction result
     * here we only consider vector with length less than 1024
    */
    __shared__ Ty reduce_smem[WarpSize];
    int warp_id = threadIdx.x / WarpSize, lane_id = threadIdx.x % WarpSize;
    Ty cur_val  = threadIdx.x < length ? vector[threadIdx.x] : 0;
    Ty result   = warp_reduce_sum(cur_val, WarpSize);
    if (lane_id == 0) reduce_smem[warp_id] = result;
    __syncthreads();
    if (threadIdx.x < WarpSize) {
        Ty cur_val = reduce_smem[threadIdx.x];
        result     = warp_reduce_sum(cur_val, WarpSize);
    }
    if (threadIdx.x == 0)
        *output = result;
}

template <typename Ty, int WarpSize = 32>
__global__ void reduce_max(const Ty* const vector, Ty* output, int length) {
    /**
     * length might be subdivided into several warps, and each warp will first have reduction result
     * here we only consider vector with length less than 1024
    */
    __shared__ Ty reduce_smem[WarpSize];
    int warp_id = threadIdx.x / WarpSize, lane_id = threadIdx.x % WarpSize;
    Ty cur_val  = threadIdx.x < length ? vector[threadIdx.x] : INT_MIN;
    Ty result   = warp_reduce_max(cur_val, WarpSize);
    if (lane_id == 0) reduce_smem[warp_id] = result;
    __syncthreads();
    if (threadIdx.x < WarpSize) {
        Ty cur_val = reduce_smem[threadIdx.x];
        result     = warp_reduce_max(cur_val, WarpSize);
    }
    if (threadIdx.x == 0)
        *output = result;
}

template <typename Ty>
void generateRandomVector(Ty* const data, int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Ty> dis(0, 1000);
    for (int i = 0; i < length; i++)
        data[i] = dis(gen);
}

template <>
void generateRandomVector<int>(int* const data, int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1000);
    for (int i = 0; i < length; i++)
        data[i] = dis(gen);
}

template <typename Ty = float, int Length = 1024>
void call_reduce_max() {
    Ty *host_data, *result;
    CUDA_CHECK_RETURN(cudaMallocManaged(&host_data, sizeof(Ty) * Length));
    CUDA_CHECK_RETURN(cudaMallocManaged(&result, sizeof(Ty)));
    generateRandomVector<Ty>(host_data, Length);
    int num_thread = next_multiple_of_32(Length);       // padding to multiple of 32

    TicToc timer;
    timer.tic();
    reduce_max<Ty><<<1, num_thread>>>(host_data, result, Length);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    printf("GPU time: %f ms\n", timer.toc());

    timer.tic();
    Ty std_out = INT_MIN;
    for (int i = 0; i < Length; i++)
        std_out = std::max(std_out, host_data[i]);
    printf("CPU time: %f ms\n", timer.toc());
    
    std::cout << "Is close: " << int((std_out - (*result)) < 1e-5f) << ". GPU: " << *result << ". CPU: " << std_out << std::endl;

    CUDA_CHECK_RETURN(cudaFree(host_data));
    CUDA_CHECK_RETURN(cudaFree(result));
}

template <typename Ty = float, int Length = 1024>
void call_reduce_sum() {
    Ty *host_data, *result;
    CUDA_CHECK_RETURN(cudaMallocManaged(&host_data, sizeof(Ty) * Length));
    CUDA_CHECK_RETURN(cudaMallocManaged(&result, sizeof(Ty)));
    generateRandomVector<Ty>(host_data, Length);
    int num_thread = next_multiple_of_32(Length);       // padding to multiple of 32

    TicToc timer;
    timer.tic();
    reduce_sum<Ty><<<1, num_thread>>>(host_data, result, Length);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    printf("GPU time: %f ms\n", timer.toc());

    timer.tic();
    Ty std_out = 0;
    for (int i = 0; i < Length; i++)
        std_out += host_data[i];
    printf("CPU time: %f ms\n", timer.toc());
    
    std::cout << "Is close: " << int((std_out - (*result)) < 1e-5f) << ". GPU: " << *result << ". CPU: " << std_out << std::endl;

    CUDA_CHECK_RETURN(cudaFree(host_data));
    CUDA_CHECK_RETURN(cudaFree(result));
}


int main() {
    call_reduce_sum<float, 32>();
    return 0;
}