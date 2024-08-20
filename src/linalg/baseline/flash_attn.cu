/**
 * The softmax implemention mentioned in 
 * FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
 * 
 * QKV attention
 * 4096 original feature size, MHA let's set 8 heads
 * therefore, QKV -> (2048 ~ 8192, 512)
 * 
 * @author: Qianyue He
 * @date:   2024.3.20
*/

#include "utils.h"
#include <iostream>
#include <cuda_runtime.h>

// TODO: float4 in CUDA? How does it work?
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

/**
 * fuse Q, K multiplication, scale and softmax all in one kernel
 * Q is pre-transposed
 * unfortunately, without V fused in here, we can't save memory
 * we still need to store Q^T K matrix, which can be pretty large
 * when the sequence is pretty long
*/
template <int BN = 128, int BK = 8, int CF = 8>
__global__ void qk_soft_max_forward_fence(
	float* __restrict__ Q, float* __restrict__ K, float* softmax, float* QK_out, int N, int d) 
{
    // step 1: Q^T @ K with scale, step 2: soft max on Q and K
    constexpr patch_rid = threadIdx.x >> 4, patch_cid = threadIdx.x % 16;
    const int gmem_q_rbase = blockIdx.y * BN * d, gmem_k_cbase = blockIdx.x * BN;
    const int g_row_num = threadIdx.x >> 1, g_col_num = threadIdx.x & 1,
              k_row_num = threadIdx.x >> 5, k_col_num = threadIdx.x % 32;
    const int th_q_rbase = g_row_num * d + gmem_q_rbase, th_k_cbase = (k_col_num << 2) + gmem_k_cbase;
    const int patch_num  = d / BK;
    __shared__ float Qpatch[BN][BK], Kpatch[BK][BN];
    float tls[CF][CF], row_reduce[CF], scale = 1.f / sqrtf(d);
    memset(tls, 0, sizeof(float) * CF * CF);

    for (int k = 0; k < patch_num; k++) {
        const int gmem_q_addr = th_q_rbase + k * BK + (g_col_num << 2), gmem_k_addr = th_k_cbase + BK * k * N;
        // copy float4 from HBM to SRAM
        FLOAT4(Qpatch[g_row_num][g_col_num << 2]) = FLOAT4(Q[gmem_q_addr]);
        FLOAT4(Kpatch[k_row_num][k_col_num << 2]) = FLOAT4(Q[gmem_k_addr]);
        __syncthreads();

        // calculate CF * CF patch result
        for (int r = 0; r < CF; r++) {
            for (int c = 0; c < CF; c++) {
                float result = 0;
                for (int k = 0; k < BK; k++) {
                    result += Qpatch[patch_rid + r][k] * Kpatch[k][patch_cid + c];
                }
                tls[r][c] += result * scale;
            }
        }
        __syncthreads();
    }
    // exp calculation before softmax / row reduce (8 * 8 -> 8 * 1)
    for (int r = 0; r < CF; r++) {
        float row_sum = 0;
        for (int c = 0; c < CF; c++) {
            float value = expf(tls[r][c]);
            tls[r][c] = value;
            row_sum += value;
        }
        row_reduce[r] = row_sum;
    }
    // row-wise summation and accumulate results to softmax row storage
    // 16 * (8 * 1) should be reduced (summed) to 1 * (8 * 1)
    for (int r = 0; r < CF; r++) {
        row_reduce[r] = warp_reduce(row_reduce[r], 8);
    } 
    // accumulate to HBM cache
    int row_id = blockIdx.y * BN + patch_rid * CF;
    if (patch_cid == 0) {       // only 16 threads, 16 * 8 = 128
        for (int r = 0; r < CF; r++)
            atomicAdd(softmax + row_id + r, row_reduce[r]);
    }
    // this will effectively stall until all the blocks run till this position...
    // blocks will be saved somewhere? including IP, registers? definitely no good...
    // softmax is accumulated
    __threadfence();
    // merged memory access
    float softmax_sums[8];
    FLOAT4(softmax_sums[0]) = FLOAT4(softmax[row_id]);
    FLOAT4(softmax_sums[4]) = FLOAT4(softmax[row_id + 4]);
    for (int r = 0; r < CF; r++) {
        float normalization = 1.f / softmax_sums[r];
        for (int c = 0; c < CF; c++)
            tls[r][c] *= normalization;
        int gmem_out_addr = (row_id + r) * N + blockIdx.x * BN + patch_cid * CF;
        FLOAT4(QK_out[gmem_out_addr])     = FLOAT4(tls[r][c]);
        FLOAT4(QK_out[gmem_out_addr + 4]) = FLOAT4(tls[r][c + 4]);
    } 
    // completed
}

/**
 * This implementation is called in a stream-multi-processing way
 * we will call this kernel multiple times and each kernel have their onw threadfence? I think?
*/
template <int BN = 128, int BK = 8, int CF = 8>
__global__ void qk_soft_max_forward_stream(
	float* __restrict__ Q, float* __restrict__ K, float* softmax, float* QK_out, int block_y, int N, int d) 
{
    // step 1: Q^T @ K with scale, step 2: soft max on Q and K
    constexpr patch_rid = threadIdx.x >> 4, patch_cid = threadIdx.x % 16;
    const int gmem_q_rbase = block_y * BN * d, gmem_k_cbase = blockIdx.x * BN;
    const int g_row_num = threadIdx.x >> 1, g_col_num = threadIdx.x & 1,
              k_row_num = threadIdx.x >> 5, k_col_num = threadIdx.x % 32;
    const int th_q_rbase = g_row_num * d + gmem_q_rbase, th_k_cbase = (k_col_num << 2) + gmem_k_cbase;
    const int patch_num  = d / BK;
    __shared__ float Qpatch[BN][BK], Kpatch[BK][BN];
    float tls[CF][CF], row_reduce[CF], scale = 1.f / sqrtf(d);
    memset(tls, 0, sizeof(float) * CF * CF);

    for (int k = 0; k < patch_num; k++) {
        const int gmem_q_addr = th_q_rbase + k * BK + (g_col_num << 2), gmem_k_addr = th_k_cbase + BK * k * N;
        // copy float4 from HBM to SRAM
        FLOAT4(Qpatch[g_row_num][g_col_num << 2]) = FLOAT4(Q[gmem_q_addr]);
        FLOAT4(Kpatch[k_row_num][k_col_num << 2]) = FLOAT4(Q[gmem_k_addr]);
        __syncthreads();

        // calculate CF * CF patch result
        for (int r = 0; r < CF; r++) {
            for (int c = 0; c < CF; c++) {
                float result = 0;
                for (int k = 0; k < BK; k++) {
                    result += Qpatch[patch_rid + r][k] * Kpatch[k][patch_cid + c];
                }
                tls[r][c] += result * scale;
            }
        }
        __syncthreads();
    }
    // exp calculation before softmax / row reduce (8 * 8 -> 8 * 1)
    for (int r = 0; r < CF; r++) {
        float row_sum = 0;
        for (int c = 0; c < CF; c++) {
            float value = expf(tls[r][c]);
            tls[r][c] = value;
            row_sum += value;
        }
        row_reduce[r] = row_sum;
    }
    // row-wise summation and accumulate results to softmax row storage
    // 16 * (8 * 1) should be reduced (summed) to 1 * (8 * 1)
    for (int r = 0; r < CF; r++) {
        row_reduce[r] = warp_reduce(row_reduce[r], 8);
    } 
    // accumulate to HBM cache
    int row_id = block_y * BN + patch_rid * CF;
    if (patch_cid == 0) {       // only 16 threads, 16 * 8 = 128
        for (int r = 0; r < CF; r++)
            atomicAdd(softmax + row_id + r, row_reduce[r]);
    }
    // this will effectively stall until all the blocks run till this position...
    // blocks will be saved somewhere? including IP, registers? definitely no good...
    // softmax is accumulated
    __threadfence();
    // merged memory access
    float softmax_sums[8];
    FLOAT4(softmax_sums[0]) = FLOAT4(softmax[row_id]);
    FLOAT4(softmax_sums[4]) = FLOAT4(softmax[row_id + 4]);
    for (int r = 0; r < CF; r++) {
        float normalization = 1.f / softmax_sums[r];
        for (int c = 0; c < CF; c++)
            tls[r][c] *= normalization;
        int gmem_out_addr = (row_id + r) * N + blockIdx.x * BN + patch_cid * CF;
        FLOAT4(QK_out[gmem_out_addr])     = FLOAT4(tls[r][c]);
        FLOAT4(QK_out[gmem_out_addr + 4]) = FLOAT4(tls[r][c + 4]);
    } 
    // completed
}



