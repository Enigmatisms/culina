#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "utils.h"

#define FLOAT4(x) *reinterpret_cast<float4*>(&x)
#define CONST_FLOAT4(x) *reinterpret_cast<const float4*>(&x)

// 蝶形运算
template <int WARP_SIZE = 32>
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int lane_mask = WARP_SIZE >> 1; lane_mask >= 1; lane_mask >>= 1) {
        val += __int_as_float(__shfl_xor_sync(0xffffffff, __float_as_int(val), lane_mask));
    }
    return val;
}

// Warp reduce 的核心实现点：
// 1. 每个 warp 加 32 个值输出一个
// 2. 一个 thread 可以处理多个数据进行 warp reduce，假设我们这里进行 coarsening = 4
// 也即一个 thread 会跨 4 次 blockDim.x
// 3. 计算结果每次保存在 shared memory 中，shared memory 大小为 32 (如果不为 32 则需要多几次 warp reduce)
template <int THREAD_NUM = 256>
__global__ void warp_reduce_sum_kernel(const float* const __restrict__ data, float* __restrict__ output, int length) {
    constexpr int shared_size = THREAD_NUM >> 5;
    int tid = threadIdx.x, nthreads = blockDim.x, index = tid;
    // 如果是针对任意大小的，需要考虑尾部的问题，warp reduce 要求参与 shfl 的每一个线程都是 active 的
    // tid 小于 length 时，可以取 global memory 进行运算，否则需要给 0
    // 一个 block 256 个线程时，每个 block 每次可以得到 8 个值 (256 大小区域)
    __shared__ float local_mem[THREAD_NUM >> 5];        // 初始化应该就是 0, THREAD_NUM / 32
    do {
        float val = index < length ? data[index] : 0;       // 可以选择 padding
        val = warp_reduce_sum(val);
        if ((tid & 31) == 0)                            // mod 32 是 0
            local_mem[tid >> 5] += val;
        index += nthreads;
    } while (__any_sync(0xffffffff, index < length));                 // 只要有线程还可以运行就继续
    // warp 可以提前退出到这里
    __syncthreads();
    if (tid < 32) {
        float result = warp_reduce_sum(tid < shared_size ? local_mem[tid] : 0);
        if (tid == 0)
            *output = result;
    }
}

__forceinline__ __device__ float sum_float4(const float4& val) {
    return val.x + val.y + val.z + val.w;
}

template <int THREAD_NUM = 256>
__global__ void block_warp_reduce_sum_kernel(const float* const __restrict__ data, float* __restrict__ output, int total_length) {
    // 本逻辑中，每个 block 处理一个 warp reduce sum 使用 atomicAdd
    // 本例子中，threads coarsening 发生在 local 位置（一次取四个，也即一个 warp 一次处理的范围为 128）
    // 这里我们不考虑边界问题，认为一定是 128 的整数倍 --> 如果是 256 threads 将会有 128 * 8 = 1024 的处理范围
    constexpr int shared_size = THREAD_NUM >> 5;
    int tid = threadIdx.x;
    int index = (tid + blockIdx.x * blockDim.x) << 2;
    __shared__ float local_mem[THREAD_NUM >> 5];        // 初始化应该就是 0, THREAD_NUM / 32

    float val = index < total_length ? sum_float4(CONST_FLOAT4(data[index])) : 0;
    val = warp_reduce_sum(val);
    if ((tid & 31) == 0)                            // mod 32 是 0
        local_mem[tid >> 5] = val;
    __syncthreads();
    if (tid < 32) {
        float result = warp_reduce_sum(tid < shared_size ? local_mem[tid] : 0);
        if (tid == 0)
            atomicAdd(output, result);
    }
}

float sum_cuda(torch::Tensor input1) {
    TORCH_CHECK(input1.is_cuda(),       "Input tensor must be a CUDA tensor");

    const int size = input1.numel();
    std::cout << "Tensor 1 with shape " << input1.sizes() << std::endl;
    float* data1 = input1.data_ptr<float>();
    float* host_device_float = nullptr;
    CUDA_CHECK_RETURN(cudaMallocManaged(&host_device_float, sizeof(float)));
    constexpr int num_threads = 1024;
    warp_reduce_sum_kernel<num_threads><<<1, num_threads>>>(data1, host_device_float, size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    float result = *host_device_float;
    CUDA_CHECK_RETURN(cudaFree(host_device_float));
    return result;
}

float block_sum_cuda(torch::Tensor input1) {
    TORCH_CHECK(input1.is_cuda(),       "Input tensor must be a CUDA tensor");

    const int size = input1.numel();
    std::cout << "Tensor 1 with shape " << input1.sizes() << std::endl;
    float* data1 = input1.data_ptr<float>();
    float* host_device_float = nullptr;
    CUDA_CHECK_RETURN(cudaMallocManaged(&host_device_float, sizeof(float)));
    constexpr int num_threads = 256;
    int num_grid = (size + num_threads - 1) / num_threads;
    block_warp_reduce_sum_kernel<num_threads><<<num_grid, num_threads>>>(data1, host_device_float, size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    float result = *host_device_float;
    CUDA_CHECK_RETURN(cudaFree(host_device_float));
    return result;
}


PYBIND11_MODULE(cuda_reduce, m) {
    m.def("sum_cuda", &sum_cuda, "CUDA sum: single block.");
    m.def("block_sum_cuda", &block_sum_cuda, "CUDA sum: multiple block.");
}
