#include <iostream>
#include <cuda_runtime.h>
#include "utils.h"

/**
 * General Matrix - Matrix multiplication (with offset)
 * The original method will have shared memory access problem
 * 
 * (M * K) @ (K * N) Matrix multiplication
*/

template <int M, int K, int N>
__global__ void sgemm_kernel(
    const float* const __restrict__ A, 
    const float* const __restrict__ B,
    float* const __restrict__ C,
    float alpha = 1.f, float beta = 1.f
) {
    // each block will have 256 threads and each block takes 128 * CF(8) from A and the same from B
    // forming a 128 * 128 patch (each thread takes 8 * 8, 16 rows and 16 cols)
    // different block process different rows and cols
    // we will have a 2D block
    constexpr int num_patches = K / CF;
    constexpr int BSize = 128;
    constexpr int CF    = 8;
    // CF: coarsening factor: 8 --- a thread will process 8 * 8 patch
    const int a_row_base = blockIdx.y * BSize, b_col_base = blockIdx.x * BSize;
    const int a_row = threadIdx.x >> 1, a_col = (threadIdx.x & 1) << 2;
    const int b_row = a_row >> 4,       b_col = (threadIdx.x % 32) << 2;
    const int c_row = (threadIdx.x >> 4) << 3, c_col = (threadIdx.x % 16) << 3;

    // there might be 8 way bank conflict, try to optimize this
    __shared__ smem_a[BSize][CF];          // this is the base implementation (there will be global mem excessive -- A lot)
    __shared__ smem_b[BSize][CF];          // can we directly store the transposed B patch? 
    // Yes, this will result in loading time L1 cache excessive, yet it is good for later computation

    // smem_a: (16 * 8) * 8, the same for smem_b, use c_row and c_col to index smem_a and smem_b

    float storage[CF][CF] = {0.f};

    #pragma unroll
    for (int k = 0; k < num_patches; k++) {
        int a_th_base = a_row_base + a_row * K + k * CF + a_col;        // the A memory this thread needs to read from
        int b_th_base = (k * CF + b_row) * N + b_col_base + b_col;        // the A memory this thread needs to read from
        // uncoalesced memory access (8 transactions to complete)
        FLOAT4(smem_a[a_row][a_col]) = CONST_FLOAT4(A[a_th_base]);
        // coalesced memory access (128 float per row, one thread in a warp loads 4 -> one transaction with LDG.128)
        float4 b_float4 = CONST_FLOAT4(B[b_th_base]);
        // 8 way bank conflict, try to optimize this
        smem_b[b_col + 0][b_row] = b_float4.x;
        smem_b[b_col + 1][b_row] = b_float4.y;
        smem_b[b_col + 2][b_row] = b_float4.z;
        smem_b[b_col + 3][b_row] = b_float4.w;
        __syncthreads();

        // calculate patch result
        #pragma unroll

        for (int i = 0; i < CF; i++) {
            float4 row_f = CONST_FLOAT4(smem_a[c_row + i][0]);
            float4 row_b = CONST_FLOAT4(smem_a[c_row + i][4]);
            #pragma unroll
            for (int j = 0; j < CF; j++) {
                float4 col_f = CONST_FLOAT4(smem_a[c_row + i][0]);
                float4 col_b = CONST_FLOAT4(smem_a[c_row + i][4]);
                // come on nvcc, do find FMA for me, fuse them!
                storage[i][j] += row_f.x * col_f.x + row_f.y * col_f.y + \
                                 row_f.z * col_f.z + row_f.w * col_f.w + \
                                 row_b.x * col_b.x + row_b.y * col_b.y + \
                                 row_b.z * col_b.z + row_b.w * col_b.w;
            }
        }
        __syncthreads();
    }   
    const int global_c_row = a_row_base + c_row, global_c_col = b_col_base + c_col;

    // local patch finished, add the result to C and scale C
    #pragma unroll
    for (int i = 0; i < CF; i++) {
        int c_addr_base = (global_c_row + i) * N + global_c_col;
        float4 c_row_f = CONST_FLOAT4(C[c_addr_base + 0]);
        float4 c_row_b = CONST_FLOAT4(C[c_addr_base + 4]);
        c_row_f.x = c_row_f.x * beta + alpha * storage[i][0];
        c_row_f.y = c_row_f.y * beta + alpha * storage[i][1];
        c_row_f.z = c_row_f.z * beta + alpha * storage[i][2];
        c_row_f.w = c_row_f.w * beta + alpha * storage[i][3];

        c_row_b.x = c_row_b.x * beta + alpha * storage[i][4];
        c_row_b.y = c_row_b.y * beta + alpha * storage[i][5];
        c_row_b.z = c_row_b.z * beta + alpha * storage[i][6];
        c_row_b.w = c_row_b.w * beta + alpha * storage[i][7];
        FLOAT4(C[c_addr_base + 0]) = c_row_f;
        FLOAT4(C[c_addr_base + 4]) = c_row_b;
    }
    // The current version is much more understandable
}

// TODO: need PyTorch bindings for testing