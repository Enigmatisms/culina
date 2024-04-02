/**
 * Implement and benchmarking SGEMM
 * GEMM is pretty difficult (and trivial) to write
 * took me... very long to finish the whole implementation (2 hours)
 * addressing is pretty nasty, draw a diagram everytime you need to do it
 * 
 * @author: Qianyue He
 * @date:   2024-3-18
*/

#include "../utils.h"
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include "../utils.h"

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT4_CONST(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

#define PINNED_MEMORY
#define OMP_THREADS 64


// One thread will process a 4 * 4 block, each block has 16 * 16 threads -> (64, 64)
// 64 blocks (16 * 16) -> (1024, 1024)
__global__ void transpose(const float* const A, float* const B, int row) {
    int row_base = blockIdx.y << 6, col_base = blockIdx.x << 6;
    int from_addr = (row_base + (threadIdx.y << 2)) * row + col_base + (threadIdx.x << 2);
    int to_addr   = (col_base + (threadIdx.x << 2)) * row + row_base + (threadIdx.y << 2);
    for (int i = 0; i < 4; i++, from_addr += row, ++to_addr) {
        // two FLOAT4 op
        float4 f1 = FLOAT4_CONST(A[from_addr]);
        B[to_addr]           = f1.x;
        B[to_addr + row]     = f1.y;
        B[to_addr + row * 2] = f1.z;
        B[to_addr + row * 3] = f1.w;
    }
}

__global__ void transpose_naive(const float* const A, float* const B, int row) {
    // same 256 threads each block, (1024 * 1024) matrix, therefore
    // gridDim.x 4, gridDim.y 1024
    int from_addr = blockIdx.y * row + (blockIdx.x << 8) + threadIdx.x;
    int to_addr   = ((blockIdx.x << 8) + threadIdx.x) * row + blockIdx.y;
    B[to_addr] = A[from_addr];
}

// Row = N, Col = N, N is the multiple of 16
template <int Row>
class SqrMatrix {
private:
    float* _data;           // 64 Bytes aligned (fit in a cacheline)
public:
    int row() const {return Row;}
    int col() const {return Row;}

    const float* data() const {return this->_data;}
    float* data() {return this->_data;}

    SqrMatrix(bool random = false) {
        _data = new float[Row * Row];
        if (!random) {
            memset(_data, 0, sizeof(float) * Row * Row);
        } else {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0, 5);
            for (int i = 0; i < Row; i++) {
                for (int j = 0; j < Row; j++) {
                    _data[i * Row + j] = dis(gen);
                }
            }
        }
    }

    ~SqrMatrix() {
        delete [] _data;
    }

    float& operator() (int i, int j) {
        return _data[i * Row + j];
    }

    float operator() (int i, int j) const {
        return _data[i * Row + j];
    }

    float& operator[] (int index) {
        return _data[index];
    }

    float operator[] (int index) const {
        return _data[index];
    }

    // tiled-based transpose
    SqrMatrix T() const {
        // CPU cacheline is 64 Bytes, therefore we tile the transpose by 16 * 16 
        SqrMatrix new_mat;
        int tile_per_row = Row >> 4;
        // this can even use multi-threading
        for (int tile_i = 0; tile_i < tile_per_row; tile_i++) {
            for (int tile_j = 0; tile_j < tile_per_row; tile_j++) {
                int base = (tile_i << 4) * Row + (tile_j << 4);
                for (int i = 0; i < 16; i++, base += Row) {
                    for (int j = 0; j < 16; j++) {
                        new_mat((tile_j << 4) + j, (tile_i << 4) + i) = \
                        _data[base + j];
                    }
                }
            }
        }
        return new_mat;
    }

    SqrMatrix T_naive() const {
        // CPU cacheline is 64 Bytes, therefore we tile the transpose by 16 * 16 
        SqrMatrix new_mat;
        // this can even use multi-threading
        for (int i = 0; i < Row; i++) {
            int base = i * Row;
            for (int j = 0; j < Row; j++) {
                new_mat(j, i) = _data[base + j];
            }
        }
        return new_mat;
    }

    SqrMatrix T_cuda(double& time) const {  
        SqrMatrix result;
        float* dev_mat = nullptr, *trans_mat = nullptr;
        size_t size = sizeof(float) * Row * Row;
        CUDA_CHECK_RETURN(cudaMalloc(&dev_mat, size));
        CUDA_CHECK_RETURN(cudaMalloc(&trans_mat, size));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_mat, _data, size, cudaMemcpyHostToDevice));

        TicToc timer;
        timer.tic();
        transpose<<<dim3(16, 16), dim3(16, 16)>>>(dev_mat, trans_mat, Row);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        time = timer.toc();

        CUDA_CHECK_RETURN(cudaMemcpy(result.data(), trans_mat, size, cudaMemcpyDeviceToHost));

        CUDA_CHECK_RETURN(cudaFree(dev_mat));
        CUDA_CHECK_RETURN(cudaFree(trans_mat));
        return result;
    }

    SqrMatrix T_cuda_naive(double& time) const {  
        SqrMatrix result;
        float* dev_mat = nullptr, *trans_mat = nullptr;
        size_t size = sizeof(float) * Row * Row;
        CUDA_CHECK_RETURN(cudaMalloc(&dev_mat, size));
        CUDA_CHECK_RETURN(cudaMalloc(&trans_mat, size));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_mat, _data, size, cudaMemcpyHostToDevice));

        TicToc timer;
        timer.tic();
        transpose_naive<<<dim3(4, 1024), dim3(256)>>>(dev_mat, trans_mat, Row);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        time = timer.toc();

        CUDA_CHECK_RETURN(cudaMemcpy(result.data(), trans_mat, size, cudaMemcpyDeviceToHost));

        CUDA_CHECK_RETURN(cudaFree(dev_mat));
        CUDA_CHECK_RETURN(cudaFree(trans_mat));
        return result;
    }
};

int main() {
    constexpr int row = 1024;
    SqrMatrix<row> mat(true);

    printf("Mat initialized.\n");
    
    TicToc timer;
    timer.tic();
    auto cpu_trans = mat.T();
    printf("CPU transpose (tiling) finished in %.5f ms.\n", timer.toc());

    timer.tic();
    auto cpu_trans_naive = mat.T_naive();
    printf("CPU transpose (naive) finished in %.5f ms.\n", timer.toc());

    // warm up
    // for (int i = 0; i < 4; i++) {
    //     double gpu_time = 0;
    //     auto gpu_trans = mat.T_cuda(gpu_time);
    //     auto gpu_trans_naive = mat.T_cuda_naive(gpu_time);
    // }

    double gpu_time = 0;
    auto gpu_trans = mat.T_cuda(gpu_time);
    printf("GPU transpose (tiling) finished in %.5f ms.\n", gpu_time);

    auto gpu_trans_naive = mat.T_cuda_naive(gpu_time);
    printf("GPU transpose (naive) finished in %.5f ms.\n", gpu_time);

    float diff_cpu = compare_result(cpu_trans.data(), cpu_trans_naive.data(), row, row);
    float diff_gpu = compare_result(cpu_trans.data(), gpu_trans.data(), row, row);
    float diff_gpu_naive = compare_result(cpu_trans.data(), gpu_trans_naive.data(), row, row);

    printf("Difference: CPU (tiling) - CPU (naive): %f\n", diff_cpu);
    printf("Difference: CPU (tiling) - GPU (tiling): %f\n", diff_gpu);
    printf("Difference: CPU (tiling) - GPU (naive): %f\n", diff_gpu_naive);
    return 0;
}