/// @author (Unknown) @copyright Unknown
#pragma once
#include <cstdio>
#include <type_traits>
#include <cuda_runtime.h>

namespace culin {

template <typename T>
struct is_numeric {
    static constexpr bool value = std::is_arithmetic_v<T> || std::is_enum_v<T>;
};

template <typename T>
inline constexpr bool is_numeric_v = is_numeric<T>::value;

__host__ static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__host__ static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	printf("%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err), err, file, line);
	exit (1);
}

}   // end namespace culin
