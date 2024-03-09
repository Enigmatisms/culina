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
        operator_##func_name<<<Row, Col, 0, streams[i]>>>(this->data(), mat.data(), new_mat->data()); \
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
        operator_##func_name<<<Row, Col, 0, streams[i]>>>(this->data(), mat.data(), this->data()); \
    } \
    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); \
    for (int i = 0; i < __stream_use; i++) \
        cudaStreamDestroy(streams[i]); \
}

/**
 * this Matrix implementation can be constructed both on CPU and GPU side
 * but I will implement GPU version first
*/
template <size_t Row, size_t Col, typename Ty = float>
class Matrix {
public:
    __device__ Matrix();
    // matrix storage is explicitly managed
    __device__ ~Matrix();
public:
    /**
     * Matrix multiplication. For element wise multiplication, please see (ewise_mul)
     * note that the output type needs to be deduced, due to that we don't know the shape of
     * mat yet. All the operations will be called on host side, while the data is stored on the GPU
     * global mem
    */
    template <typename MatType>
    __host__ auto operator*(MatType&& mat) {
        
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
private:
    // whether the current data is stored on GPU
    bool gpu_managed;
    // data field (1D)
    Ty* data;
};

}   // end namespace culin
