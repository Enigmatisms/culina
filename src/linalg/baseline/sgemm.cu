/**
 * Implement and benchmarking SGEMM
 * GEMM is pretty difficult (and trivial) to write
 * took me... very long to finish the whole implementation (2 hours)
 * addressing is pretty nasty, draw a diagram everytime you need to do it
 * 
 * @author: Qianyue He
 * @date:   2024-3-18
*/

#include "utils.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// TODO: float4 in CUDA? How does it work?
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define PINNED_MEMORY
#define OMP_THREADS 64
#define COARSEN_4

/**
 * alpha * AB + beta * c
 * Here is the tiling method
 * 
 *   | --------------Y---------------
 *   | 	   K          N          N
 *   |  |     |    |     |    |     |
 *   X M|     | * K|     | + M|     |
 *   |  |     |    |     |    |     |
 *   |     A          B          C
 * 
 * Holy, shit... I have done it... no minor feat
*/ 

template <int BM, int BN, int BK = 8, int CF = 8>
__global__ void sgemm_kernel(
	float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, float alpha, float beta, 
	int m, int n, int k) {
	// tile size: 128 * 128 (for a block), a block contains 16 * 16 threads (a thread process 8 * 8 patch)
	// thread coarsening. Note that for direction K, we will continue to use 128 as tiling size
	// BM: tile size (M), BN: tile size: N, BK: tile size K
	// CF: coarsening factor = 8
	// FIXME: I don't want to consider padding here! M, N, K will be the multiple of 128
	__align__(32) __shared__ float tile_a[BM][BK];
	__align__(32) __shared__ float tile_b[BK][BN];				// transposed store
	__align__(16) float local_c[CF][CF];											// how the thread local storage is allocated? using register?
	
	// [step 1]: copy global mem to smem, 16 * 16 thread, 128 * 128 address: 4 for each, use FLOAT4
	// smem 2D address
	int gmem_base_a = (blockIdx.x << 7) * k, gmem_base_b = blockIdx.y << 7;
	int rbase = (threadIdx.x >> 4) << 3, cbase = (threadIdx.x % 16) << 3;

	// locate the memory address in C (to be stored to, and load from (beta scaling))
	int gmem_addr_c = ((blockIdx.x * n + blockIdx.y) << 7) + rbase * n + cbase;

	for (int i = 0; i < CF; i++) {
		int gmem_addr_ci = gmem_addr_c + i * n;
		FLOAT4(local_c[i][0]) = FLOAT4(C[gmem_addr_ci]);
		FLOAT4(local_c[i][4]) = FLOAT4(C[gmem_addr_ci + 4]);
		for (int j = 0; j < CF; j++)
			local_c[i][j] *= beta;
	}
	
	// for k / BK patches in a row-patch / col-patch
	for (int kid = 0; kid < k / BK; kid ++) {
		int tile_r = threadIdx.x >> 1, tile_c = (threadIdx.x & 1) << 2;
		int gmem_addr = gmem_base_a + (kid << 3) + tile_r * k + tile_c;			// 128 * blockIdx.x is row, (kid << 3) is column
		// load A patch
		FLOAT4(tile_a[tile_r][tile_c]) = FLOAT4(A[gmem_addr]);
		tile_r = threadIdx.x >> 5;
		tile_c = (threadIdx.x % 32) << 2;
		gmem_addr = gmem_base_b + ((kid << 3) + tile_r) * n + tile_c;
		// load B patch: this might get improved by making it storing in a transposed way
		FLOAT4(tile_b[tile_r][tile_c]) = FLOAT4(B[gmem_addr]);
		__syncthreads();

		// [step 2]: thread compute and store to local_c (register usage contained), with thread coarsening
		// compute CF * CF (8 * 8 in our case) and store to local_c
		for (int i = 0; i < CF; i++) {
			for (int j = 0; j < CF; j++) {
				float sum = 0;
				#pragma unroll
				for (int p = 0; p < BK; p++) {
					sum = fmaf(tile_a[rbase + i][p], tile_b[p][cbase + j], sum);
				}
				local_c[i][j] += sum * alpha;			// mult by the scaling factor
			}
		}
		__syncthreads();
	}

	// [step 3]: write back to C
	#pragma unroll
	for (int i = 0; i < CF; i++) {
		int gmem_addr_ci = gmem_addr_c + i * n;
		FLOAT4(C[gmem_addr_ci])     = FLOAT4(local_c[i][0]);
		FLOAT4(C[gmem_addr_ci + 4]) = FLOAT4(local_c[i][4]);
	} 
}

template <int BM = 64, int BN = 64, int BK = 8>
__global__ void sgemm_kernel_coarse4(
	float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, float alpha, float beta, 
	int m, int n, int k) {
	// tile size: 64 * 64 (for a block), a block contains 16 * 16 threads (a thread process 4 * 4 patch)
	// BM: tile size (M), BN: tile size: N, BK: tile size K
	__align__(32) __shared__ float tile_a[BM][BK];
	__align__(32) __shared__ float tile_b[BK][BN];				// transposed store
	__align__(16) float local_c[4][4];											// how the thread local storage is allocated? using register?
	
	// [step 1]: copy global mem to smem, 16 * 16 thread, 64 * 64 address: 4 for each, use FLOAT4
	// smem 2D address
	int gmem_base_a = (blockIdx.x << 6) * k, gmem_base_b = blockIdx.y << 6;
	int rbase = (threadIdx.x >> 4) << 2, cbase = (threadIdx.x % 16) << 2;

	// locate the memory address in C (to be stored to, and load from (beta scaling))
	int gmem_addr_c = ((blockIdx.x * n + blockIdx.y) << 6) + rbase * n + cbase;

	for (int i = 0; i < 4; i++) {
		int gmem_addr_ci = gmem_addr_c + i * n;
		FLOAT4(local_c[i][0]) = FLOAT4(C[gmem_addr_ci]);
		for (int j = 0; j < 4; j++)
			local_c[i][j] *= beta;
	}
	
	// for k / BK patches in a row-patch / col-patch: (64, 8)
	for (int kid = 0; kid < k / BK; kid ++) {
		if (threadIdx.x < 128) {
			const int tile_ra = threadIdx.x >> 1, tile_ca = (threadIdx.x & 1) << 2;
			const int gmem_addra = gmem_base_a + (kid << 3) + tile_ra * k + tile_ca;			// 128 * blockIdx.x is row, (kid << 3) is column
			const int tile_rb = threadIdx.x >> 4, tile_cb = (threadIdx.x % 16) << 2;
			const int gmem_addrb = gmem_base_b + ((kid << 3) + tile_rb) * n + tile_cb;
			// load A / B patch
			FLOAT4(tile_a[tile_ra][tile_ca]) = FLOAT4(A[gmem_addra]);
			FLOAT4(tile_b[tile_rb][tile_cb]) = FLOAT4(B[gmem_addrb]);
		}
		__syncthreads();

		// [step 2]: thread compute and store to local_c (register usage contained), with thread coarsening
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				float sum = 0;
				#pragma unroll
				for (int p = 0; p < BK; p++) {
					sum = fmaf(tile_a[rbase + i][p], tile_b[p][cbase + j], sum);
				}
				local_c[i][j] += sum * alpha;			// mult by the scaling factor
			}
		}
		__syncthreads();
	}

	// [step 3]: write back to C
	#pragma unroll
	for (int i = 0; i < 4; i++)
		FLOAT4(C[gmem_addr_c + i * n])     = FLOAT4(local_c[i][0]);
}

void sgemm_host_caller(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, float alpha, float beta, int M = 4096, int N = 2048, int K = 512) {
	float *devA, *devB, *devC, *devD;
	CUDA_CHECK_RETURN(cudaMalloc(&devA, sizeof(float) * M * K));
	CUDA_CHECK_RETURN(cudaMalloc(&devB, sizeof(float) * N * K));
	CUDA_CHECK_RETURN(cudaMalloc(&devC, sizeof(float) * M * N));
	CUDA_CHECK_RETURN(cudaMalloc(&devD, sizeof(float) * M * N));

	CUDA_CHECK_RETURN(cudaMemcpy(devA, A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devB, B, sizeof(float) * K * N, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devD, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);
	float cublas_alpha = 2.0;
	float cublas_beta = 0.5;
	cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, devA, N, devB, K, &cublas_beta, devD, N);

	// no stream
	#ifdef COARSEN_4
	dim3 grid(M / 64, N / 64);
	sgemm_kernel_coarse4<64, 64, 8><<<grid, 256>>>(devA, devB, devC, alpha, beta, M, N, K);
	#else
	dim3 grid(M / 128, N / 128);
	sgemm_kernel<128, 128, 8, 8><<<grid, 256>>>(devA, devB, devC, alpha, beta, M, N, K);
	#endif // COARSEN_4

	// implicit synchronize
	CUDA_CHECK_RETURN(cudaMemcpy(C, devC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(devA));
	CUDA_CHECK_RETURN(cudaFree(devB));
	CUDA_CHECK_RETURN(cudaFree(devC));
	CUDA_CHECK_RETURN(cudaFree(devD));
}

void sgemm_cpu_multi_threading(
	float* A, float* B, float* C, 
	float alpha, float beta, const int M, const int N, const int K
) {
	#pragma omp parallel for num_threads(OMP_THREADS)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = C[m * N + n] * beta + psum * alpha;
        }
    }
}

int main() {
	omp_set_num_threads(OMP_THREADS);
	int M = 4096, N = 2048, K = 1024;

	float alpha = 2.f, beta = 0.5f;
	float *A, *B, *C, *D;

	#ifdef PINNED_MEMORY
		CUDA_CHECK_RETURN(cudaMallocHost(&A, sizeof(float) * M * K));
		CUDA_CHECK_RETURN(cudaMallocHost(&B, sizeof(float) * K * N));
		CUDA_CHECK_RETURN(cudaMallocHost(&C, sizeof(float) * M * N));
		CUDA_CHECK_RETURN(cudaMallocHost(&D, sizeof(float) * M * N));
	#else
		A = new float [M * K];
		B = new float [N * K];
		C = new float [M * N];
		D = new float [M * N];
	#endif

	printf("Generating random matrix A...\n");
	generate_random_matrix(A, M, K);
	printf("Generating random matrix B...\n");
	generate_random_matrix(B, K, N);
	printf("Generating random matrix C and D...\n");
	generate_random_matrix(C, M, N);
	memcpy(D, C, sizeof(float) * M * N);

	TicToc timer;

	printf("CPU SGEMM calculating.\n");
	timer.tic();
	sgemm_cpu_multi_threading(A, B, C, alpha, beta, M, N, K);
	float cpu_time_ms = timer.toc();
	printf("CPU SGEMM finished in %.5f ms.\n", cpu_time_ms);
	nan_checker(C, M, N);

	printf("GPU SGEMM calculating.\n");
	timer.tic();
	sgemm_host_caller(A, B, D, alpha, beta, M, N, K);
	float gpu_time_ms = timer.toc();
	printf("GPU SGEMM finished in %.5f ms.\n", gpu_time_ms);
	nan_checker(D, M, N);

	float diff = compare_result(C, D, M, N);
	printf("GPU SGEMM: %f ms, CPU %d threads: %f ms. MAE: %.5f\n", gpu_time_ms, OMP_THREADS, cpu_time_ms, diff);


	

	#ifdef PINNED_MEMORY
		CUDA_CHECK_RETURN(cudaFreeHost(A));
		CUDA_CHECK_RETURN(cudaFreeHost(B));
		CUDA_CHECK_RETURN(cudaFreeHost(C));
		CUDA_CHECK_RETURN(cudaFreeHost(D));
	#else
		delete [] A;
		delete [] B;
		delete [] C;
		delete [] D;
	#endif
}
