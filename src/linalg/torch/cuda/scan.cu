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
template <int NTHREAD = 1024>
__global__ void naive_scan_long_kernel(const float* const __restrict__ data, float* __restrict__ output, int length) {
    constexpr int FINAL_SIZE = NTHREAD >> 1;
    const int tid = threadIdx.x + blockIdx.x * NTHREAD;
    if (tid < length)
        output[tid] = data[tid];
    for (int idle_end = 1; idle_end <= FINAL_SIZE; idle_end <<= 1) {
        if (tid >= idle_end && tid < length) {
            float prev_val = output[tid - idle_end], curr_val = output[tid];
            output[tid] = curr_val + prev_val;
        }
    }
}

template <int NTHREAD = 1024>
__global__ void naive_block_scan_kernel(float* output, int num_blocks, int length) {
    const int tid = threadIdx.x;
    for (int i = 1; i < num_blocks; i++) {
        int curr_addr = NTHREAD * i;
        float val = output[curr_addr - 1];
        curr_addr += tid;
        if (curr_addr < length) {
            output[curr_addr] += val;
        }
    }
}

torch::Tensor naive_cumsum_long(torch::Tensor input1) {
    TORCH_CHECK(input1.is_cuda(),       "Input tensor must be a CUDA tensor");

    const int size = input1.numel();
    torch::Tensor output = torch::zeros({size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    constexpr int num_threads = 1024;
    const int num_blocks = (size + num_threads - 1) / num_threads;
    const float* const data1 = input1.data_ptr<float>();
    float* const output_ptr = output.data_ptr<float>();
    naive_scan_long_kernel<num_threads><<<num_blocks, num_threads>>>(data1, output_ptr, size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    naive_block_scan_kernel<num_threads><<<1, num_threads>>>(output_ptr, num_blocks, size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return output;
}

PYBIND11_MODULE(cuda_scan, m) {
    m.def("naive_cumsum_long", &naive_cumsum_long, "CUDA cumsum: for long tensor.");
}
