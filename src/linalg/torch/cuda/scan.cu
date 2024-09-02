/**
 * @file scan.cu
 * @author Qianyue He
 * @brief Parallel Scan (prefix sum - cumsum)
 * @date 2024-09-02
 * @copyright Copyright (c) 2024
 */

#include <iostream>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include "utils.h"

#define FLOAT4(x) *reinterpret_cast<float4*>(&x)
#define CONST_FLOAT4(x) *reinterpret_cast<const float4*>(&x)

/**
 * @brief naive_scan
 * 有这么几个要点：每个 block 只能 scan 自己内部的值
 * 每个 block 计算完成后，最后会有单个 kernel 进行一次 warp scan
 * 一个 block 内部的处理：蝶形运算
 */
template <uint32_t NTHREAD = 1024>
__global__ void naive_scan_long_kernel(const float* const __restrict__ data, float* __restrict__ output, uint32_t length) {
    constexpr uint32_t FINAL_SIZE = NTHREAD >> 1;
    const uint32_t tid = threadIdx.x + blockIdx.x * NTHREAD;
    if (tid < length)
        output[tid] = data[tid];
    __syncthreads();
    for (uint32_t idle_end = 1; idle_end <= FINAL_SIZE; idle_end <<= 1) {
        // 注意，每一次读取，都需要保证不会被别的线程影响
        float cur_val  = tid < length ? output[tid] : 0;
        float prev_val = (tid - idle_end < length && threadIdx.x >= idle_end) ? output[tid - idle_end] : 0;
        __syncthreads();
        if (tid < length) {
            output[tid] = prev_val + cur_val;
        }
        __syncthreads();
    }
}

template <uint32_t NTHREAD = 1024>
__global__ void naive_block_scan_kernel(float* output, int num_blocks, uint32_t length) {
    for (uint32_t i = 1; i < num_blocks; i++) {
        uint32_t curr_addr = NTHREAD * i;
        float val = output[curr_addr - 1];
        curr_addr += threadIdx.x;
        if (curr_addr < length) {
            output[curr_addr] += val;
        }
        __syncthreads();
    }
}

template <uint32_t NTHREAD = 1024>
__global__ void shared_naive_scan_long_kernel(const float* const __restrict__ data, float* __restrict__ output, uint32_t length) {
    constexpr uint32_t FINAL_SIZE = NTHREAD >> 1;
    const uint32_t tid = threadIdx.x + blockIdx.x * NTHREAD;
    __shared__ float reducing[NTHREAD];
    if (tid < length)
        reducing[threadIdx.x] = data[tid];
    __syncthreads();
    for (uint32_t idle_end = 1; idle_end <= FINAL_SIZE; idle_end <<= 1) {
        // 注意，每一次读取，都需要保证不会被别的线程影响
        float cur_val  = tid < length ? reducing[threadIdx.x] : 0;
        float prev_val = (tid - idle_end < length && threadIdx.x >= idle_end) ? reducing[threadIdx.x - idle_end] : 0;
        __syncthreads();
        if (tid < length) {
            reducing[threadIdx.x] = prev_val + cur_val;
        }
        __syncthreads();
    }
    if (tid < length)
        output[tid] = reducing[threadIdx.x];
}

// 关于 Blelloch scan 算法，见：https://people.cs.pitt.edu/~bmills/docs/teaching/cs1645/lecture_scan.pdf
// 非常妙而且优雅
// 以下这两种方法可以记录一下，用 warp 进行的 branch less 操作
// 应该会存在一个 SEL 指令或者 predicate register，用于避免分支
// upsweep（reduce 过程）中，每次只有满足某个条件的线程才会reduce warp sync 的值，其他的线程保留原来的值
__device__ void warp_blelloch_scan_upsweep(float* data) {
    // up sweep 最多进行  16 次，2，4，8，16
    int tid = threadIdx.x, addr = threadIdx.x + blockIdx.x;
    float val = data[addr];
    #pragma unroll
    for (int mask = 1; mask <= 16; ) {
        float local_v = __int_as_float(__shfl_xor_sync(0xffffffff, __float_as_int(val), mask));
        mask <<= 1;
        val = (tid & (mask - 1)) == mask - 1 ? val + local_v : val;
    }
    data[addr] = val;
}

// downsweep 过程中，存在两个条件（下面称主要和次要）
// 满足主要条件的，将会reduce。而不满足主要却满足次要条件的，会发生交换。其他的保留原值不变
// 最后注意 exclusive scan 和 inclusive scan
__device__ void warp_blelloch_scan_downsweep(const float* const origin, float* data) {
    int tid = threadIdx.x, addr = threadIdx.x + blockIdx.x;
    float val = data[addr];
    #pragma unroll
    for (int mask = 32; mask >= 2; ) {
        int lane_mask = mask >> 1;
        float local_v = __int_as_float(__shfl_xor_sync(0xffffffff, __float_as_int(val), lane_mask));
        bool main_cond = (tid & (mask - 1)) == mask - 1,
             secd_cond = (tid & (lane_mask - 1)) == lane_mask - 1;
        val += main_cond ? local_v : 0;                     // 主要条件满足时，reduce
        val  = !main_cond && secd_cond ? local_v : val;     // 非主要条件满足，但次要条件满足时，交换
    }
    data[addr] = val + origin[addr];                        // inclusive scan (最后一个元素被包括，如果不包括，则不需要加 origin)
}

torch::Tensor naive_cumsum(torch::Tensor input1) {
    TORCH_CHECK(input1.is_cuda(),       "Input tensor must be a CUDA tensor");

    const int size = input1.numel();
    torch::Tensor output = torch::zeros({size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    constexpr int num_threads = 1024;
    const int num_blocks = (size + num_threads - 1) / num_threads;
    const float* const data1 = input1.data_ptr<float>();
    float* const output_ptr = output.data_ptr<float>();
    naive_scan_long_kernel<num_threads><<<num_blocks, num_threads>>>(data1, output_ptr, size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    if (num_blocks > 1) {
        naive_block_scan_kernel<num_threads><<<1, num_threads>>>(output_ptr, num_blocks, size);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
    return output;
}

torch::Tensor shared_naive_cumsum(torch::Tensor input1) {
    TORCH_CHECK(input1.is_cuda(),       "Input tensor must be a CUDA tensor");

    const int size = input1.numel();
    torch::Tensor output = torch::zeros({size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    constexpr int num_threads = 1024;
    const int num_blocks = (size + num_threads - 1) / num_threads;
    const float* const data1 = input1.data_ptr<float>();
    float* const output_ptr = output.data_ptr<float>();
    shared_naive_scan_long_kernel<num_threads><<<num_blocks, num_threads>>>(data1, output_ptr, size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    if (num_blocks > 1) {
        naive_block_scan_kernel<num_threads><<<1, num_threads>>>(output_ptr, num_blocks, size);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
    return output;
}

PYBIND11_MODULE(cuda_scan, m) {
    m.def("naive_cumsum", &naive_cumsum, "CUDA cumsum: Naive");
    m.def("shared_naive_cumsum", &shared_naive_cumsum, "CUDA cumsum: Naive with shared memory");
}
