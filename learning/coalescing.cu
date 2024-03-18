#include <iostream>

__host__ static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__host__ static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	printf("%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err), err, file, line);
	exit (1);
}

__global__ void coalesced_access(float* data) {
    int index = threadIdx.x; 
    #pragma unroll
    for (int i = 0; i < 4; i++, index += 256)
        ++ data[index];
}

__global__ void non_coalesced_access(float* data) {
    int index = 4 * threadIdx.x; 
    #pragma unroll
    for (int i = 0; i < 4; i++)
        ++ data[index + i];
}

int main() {
    float *g_data1, *g_data2;
    CUDA_CHECK_RETURN(cudaMalloc(&g_data1, sizeof(float) * 1024));
    CUDA_CHECK_RETURN(cudaMalloc(&g_data2, sizeof(float) * 1024));

    coalesced_access<<<1, 256>>>(g_data1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    non_coalesced_access<<<1, 256>>>(g_data2);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaFree(g_data1));
    CUDA_CHECK_RETURN(cudaFree(g_data2));
    return 0;
}