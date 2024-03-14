/**
 * culin::Matrix<N, M, type> definition
 * 
 * this matrix implementation will consider both case:
 * - matrix is small (for ordinary 2d/3d transformation computation)
 * - matrix is huge  (for image processing/massive scale matrix operation)
*/

#pragma once
#include "utils.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

namespace culin {

static constexpr int TILE_SZ   = 16;

// ================== SFINAE ====================
template<size_t Row, size_t Col, typename MatType>
struct is_matching_matrix {
    static constexpr bool value = false;
};

template<size_t Row, size_t Col, typename Ty>
struct is_matching_matrix<Row, Col, Matrix<Row, Col, Ty>> {
    static constexpr bool value = true;
};

template<typename Ty>
struct is_logical_allowed {
    static constexpr bool value = std::is_integral_v<Ty> || std::is_same_v<Ty, bool>;
};

// matrix traits
template <typename T>
struct MatrixTraits {
    static constexpr size_t numRows = 0;
    static constexpr size_t numCols = 0;
    using ElementType = void;
};

template <size_t Row, size_t Col, typename Ty>
struct MatrixTraits<Matrix<Row, Col, Ty>> {
    static constexpr size_t numRows = Row;
    static constexpr size_t numCols = Col;
    using ElementType = Ty;
};


#define ENABLE_IF_LOGICAL(Ty, enable) typename std::enable_if_t<enable || is_logical_allowed<Ty>::value>* = nullptr
// ================== SFINAE ====================

static constexpr int __stream_use = 8;

// __global__ operator generator
#define OPERATOR_OPS(op, func_name, Ty) \
    __global__ void operator_##func_name(const Ty* const src1, const Ty* const src2, Ty* const dst, size_t col) { \
        const size_t col_id = threadIdx.x, flat_index = blockIdx.x * col + col_id; \
        if (col_id < col) { \
            dst[flat_index] = src1[flat_index] op src2[flat_index]; \
        } \
    }

OPERATOR_OPS(+, add, float)
OPERATOR_OPS(-, sub, float)
OPERATOR_OPS(+, sub, int)
OPERATOR_OPS(-, sub, int)

OPERATOR_OPS(&, and, int)
OPERATOR_OPS(|, or,  int)
OPERATOR_OPS(^, xor, int)
OPERATOR_OPS(&, and, bool)
OPERATOR_OPS(|, or,  bool)
OPERATOR_OPS(^, xor, bool)

// multiplcation is not included here
/**
 * `enable` should be true, when operator is + and - (operator will apply for float / int)
 * when enable is false, ENABLE_IF_LOGICAL(enable, Ty) will only be true, if
 * Ty is of integer type or bool type. i.e, for float and double, this operation will not
 * be generated due to SFINAE
*/ 
#define BINARY_OPS_HELPER(op, func_name, Row, Col, Ty, enable) \
template <typename MatType, ENABLE_IF_LOGICAL(enable, Ty)> \
__host__ Matrix<Row, Col, Ty> operator##op(MatType&& mat) { \
    static_assert(is_numeric_v<MatType> || \
        is_matching_matrix<Row, Col, typename std::decay_t<MatType>>::value, \
        "Operand must be a numeric type or a matrix with matching dimensions" \
    ); \
    cudaStream_t streams[__stream_use]; \
    for (short i = 0; i < __stream_use; i++) { \
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking); \
    } \
    for (int i = 0; i < __stream_use; i++) { \
        Matrix<Row, Col, Ty> new_mat(); \
        operator_##func_name<<<Row, Col, 0, streams[i]>>>(this->data(), mat.data(), new_mat->data_mut()); \
    } \
    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); \
    for (int i = 0; i < __stream_use; i++) \
        cudaStreamDestroy(streams[i]); \
}

// enable should be true, when operator is + and - (operator will apply for float / int)
#define BINARY_OPS_INPLACE_HELPER(op, func_name, Row, Col, Ty, enable) \
template <typename MatType, ENABLE_IF_LOGICAL(enable, Ty)> \
__host__ void operator##op(MatType&& mat) { \
    static_assert(is_numeric_v<MatType> || \
        is_matching_matrix<Row, Col, typename std::decay_t<MatType>>::value, \
        "Operand must be a numeric type or a matrix with matching dimensions" \
    ); \
    cudaStream_t streams[__stream_use]; \
    for (short i = 0; i < __stream_use; i++) { \
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking); \
    } \
    for (int i = 0; i < __stream_use; i++) { \
        operator_##func_name<<<Row, Col, 0, streams[i]>>>(this->data_mut(), mat.data(), this->data_mut()); \
    } \
    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); \
    for (int i = 0; i < __stream_use; i++) \
        cudaStreamDestroy(streams[i]); \
}

/**
 * Tiled matrix multiplication implementation
 * When the matrix is small, matmul is bandwidth bounded
 * in this function, we tile (and implicitly pad) the matrix by 16 * 16 patches
 * this function does not support inplace matrix multiplication for now
*/
template <typename Ty>
__global__ void mat_mult(const Ty* const src1, const Ty* const src2, Ty* const dst, int src1_m, int src2_n, int k) {
    // shared_memory is initialized to be zero
    __shared__ Ty src1_tile[TILE_SZ * TILE_SZ * 2];
    Ty* src2_tile = &src1_tile[TILE_SZ * TILE_SZ];
    int blr_1 = blockIdx.x * TILE_SZ, blc_2 = blockIdx.y * TILE_SZ, bl_idx = blockIdx.z * TILE_SZ;   // base
    int base_1 = blr_1 * k + bl_idx, base_2 = bl_idx * src2_n + blc_2;
    // src1, src2 global memory load address
    int tpos_1 = base_1 + threadIdx.x + threadIdx.y * k, 
        tpos_2 = base_2 + threadIdx.x + threadIdx.y * src2_n,
        tile_y = TILE_SZ * threadIdx.y, tile_x = threadIdx.x * TILE_SZ,
        sh_pos = threadIdx.x + TILE_SZ * threadIdx.y;    
    bool valid_r = blr_1 + threadIdx.y < src1_m, valid_c = blc_2 + threadIdx.y < src2_n;
    if (valid_r && bl_idx + threadIdx.x < k)
        src1_tile[threadIdx.x + tile_y] = src1[tpos_1];
    if (valid_c && bl_idx + threadIdx.x < k)
        src2_tile[threadIdx.y + tile_x] = src1[tpos_2];      // transpose store
    // make sure global to shared is completed for all threads, otherwise the result might be erroneous
    __syncthreads();
    Ty sum = 0;
    for (int i = 0; i < TILE_SZ; i++)
        sum += src1_tile[tile_y + i] * src2_tile[tile_x + i];
    if (valid_r && valid_c)
        dst[(blr_1 + threadIdx.y) * src2_n + blc_2 + threadIdx.x] += sum;
    // if there is no sync threads, there might be data race between blocks
    // since different blocks can access the same global memory address
    __syncthreads();
}   

/**
 * this Matrix implementation can be constructed both on CPU and GPU side
 * but I will implement GPU version first
*/
template <size_t Row, size_t Col, typename Ty = float>
class Matrix {
public:
    __host__ Matrix();
    // matrix storage is explicitly managed
    __host__ ~Matrix();
public:
    /**
     * Matrix multiplication. For element wise multiplication, please see (ewise_mul)
     * note that the output type needs to be deduced, due to that we don't know the shape of
     * mat yet. All the operations will be called on host side, while the data is stored on the GPU
    */
    template <typename MatType>
    __host__ auto operator*(MatType&& mat) {
        // allocate three chunks of global mem
        static_assert(MatrixTraits<std::decay_t<MatType>>::numRows == Col, "Matrix multiplication should have matched input shapes.");
        constexpr size_t numCols = MatrixTraits<std::decay_t<MatType>>::numCols;
        Ty *src1, *src2, *dst;

        Matrix<Row, numCols> output;

        CUDA_CHECK_RETURN(cudaMalloc(&src1, sizeof(Ty) * Row * Col));
        CUDA_CHECK_RETURN(cudaMalloc(&src2, sizeof(Ty) * Col * numCols));
        CUDA_CHECK_RETURN(cudaMalloc(&dst,  sizeof(Ty) * Row * numCols));

        CUDA_CHECK_RETURN(cudaMemcpy(src1, data, sizeof(Ty) * Row * Col, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(src2, mat.data(), sizeof(Ty) * Col * numCols, cudaMemcpyHostToDevice));

        // manage streams and call the kernel function 

        CUDA_CHECK_RETURN(cudaMemcpy(output.data_mut(), dst, sizeof(Ty) * Col * numCols, cudaMemcpyDeviceToHost));
        // free three chunks of global mem
        CUDA_CHECK_RETURN(cudaFree(src1));
        CUDA_CHECK_RETURN(cudaFree(src2));
        CUDA_CHECK_RETURN(cudaFree(dst));
    }

    BINARY_OPS_HELPER(+, add, Row, Col, Ty, true)
    BINARY_OPS_HELPER(-, sub, Row, Col, Ty, true)
    BINARY_OPS_HELPER(&, and, Row, Col, Ty, false)
    BINARY_OPS_HELPER(|, or,  Row, Col, Ty, false)
    BINARY_OPS_HELPER(^, xor, Row, Col, Ty, false)

    BINARY_OPS_INPLACE_HELPER(+, add, Row, Col, Ty, true)
    BINARY_OPS_INPLACE_HELPER(-, sub, Row, Col, Ty, true)
    BINARY_OPS_INPLACE_HELPER(&, and, Row, Col, Ty, false)
    BINARY_OPS_INPLACE_HELPER(|, or,  Row, Col, Ty, false)
    BINARY_OPS_INPLACE_HELPER(^, xor, Row, Col, Ty, false)

    // allocate CPU mem, copy memory from GPU to CPU then free GPU mem
    __host__ void to_cpu();

    // allocate GPU mem, copy memory from CPU to GPU then free CPU mem
    __host__ void to_gpu();

    __host__ Ty* data_mut() {
        return this->_data;
    }

    __host__ const Ty* data() const {
        return this->_data;
    }
private:
    // whether the current data is stored on GPU
    bool gpu_managed;
    // data field (1D)
    Ty* _data;
};

}   // end namespace culin
