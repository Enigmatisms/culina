/**
 * Implement and benchmarking SMATMUL
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

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

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
__global__ void smatmul_kernel(
	float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, const int m, const int n, const int k) {
	// tile size: 128 * 128 (for a block), a block contains 16 * 16 threads (a thread process 8 * 8 patch)
	// thread coarsening. Note that for direction K, we will continue to use 128 as tiling size
	// BM: tile size (M), BN: tile size: N, BK: tile size K
	// CF: coarsening factor = 8
	// FIXME: I don't want to consider padding here! M, N, K will be the multiple of 128
	__shared__ float tile_a[BM][BK];
	__shared__ float tile_b[BK][BN];				// transposed store
	float local_c[CF][CF] = {0.f};											// how the thread local storage is allocated? using register?
	
	// [step 1]: copy global mem to smem, 16 * 16 thread, 128 * 128 address: 4 for each, use FLOAT4
	// smem 2D address
	int gmem_base_a = (blockIdx.x << 7) * k, gmem_base_b = blockIdx.y << 7;
	int rbase = (threadIdx.x >> 4) << 3, cbase = (threadIdx.x % 16) << 3;

	// locate the memory address in C (to be stored to, and load from (beta scaling))
	int gmem_addr_c = ((blockIdx.x * n + blockIdx.y) << 7) + rbase * n + cbase;

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
		#pragma unroll
		for (int i = 0; i < CF; i++) {
			#pragma unroll
			for (int j = 0; j < CF; j++) {
				#pragma unroll
				for (int p = 0; p < BK; p++) {
					local_c[i][j] += tile_a[rbase + i][p] * tile_b[p][cbase + j];
				}
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
__global__ void smatmul_kernel_coarse4(
	float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, const int m, const int n, const int k
) {
	// tile size: 64 * 64 (for a block), a block contains 16 * 16 threads (a thread process 4 * 4 patch)
	// BM: tile size (M), BN: tile size: N, BK: tile size K
	__shared__ float tile_a[BM][BK];
	__shared__ float tile_b[BK][BN];				// transposed store
	float local_c[4][4] = {0.f};											// how the thread local storage is allocated? using register?
	
	// [step 1]: copy global mem to smem, 16 * 16 thread, 64 * 64 address: 4 for each, use FLOAT4
	// smem 2D address
	int gmem_base_a = (blockIdx.x << 6) * k, gmem_base_b = blockIdx.y << 6;
	int rbase = (threadIdx.x >> 4) << 2, cbase = (threadIdx.x % 16) << 2;

	// locate the memory address in C (to be stored to, and load from (beta scaling))
	int gmem_addr_c = ((blockIdx.x * n + blockIdx.y) << 6) + rbase * n + cbase;

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

		#pragma unroll
		for (int i = 0; i < 4; i++) {
			#pragma unroll
			for (int j = 0; j < 4; j++) {
				#pragma unroll
				for (int p = 0; p < BK; p++) {
					local_c[i][j] += tile_a[rbase + i][p] * tile_b[p][cbase + j];
				}
			}
		}
		__syncthreads();
	}

	// [step 3]: write back to C
	#pragma unroll
	for (int i = 0; i < 4; i++)
		FLOAT4(C[gmem_addr_c + i * n]) = FLOAT4(local_c[i][0]);
}

/**
 * No bank conflict version from https://zhuanlan.zhihu.com/p/657632577
 * Having no clue why this is so yet.
 * 
*/
template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void sgemm_V2(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}


void smatmul_host_caller(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M = 4096, int N = 2048, int K = 512) {
	float *devA, *devB, *devC;
	CUDA_CHECK_RETURN(cudaMalloc(&devA, sizeof(float) * M * K));
	CUDA_CHECK_RETURN(cudaMalloc(&devB, sizeof(float) * N * K));
	CUDA_CHECK_RETURN(cudaMalloc(&devC, sizeof(float) * M * N));
	// CUDA_CHECK_RETURN(cudaMalloc(&devD, sizeof(float) * M * N));

	CUDA_CHECK_RETURN(cudaMemcpy(devA, A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devB, B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

	// cublasHandle_t cublas_handle;
	// cublasCreate(&cublas_handle);
	// float cublas_alpha = 1.0;
	// float cublas_beta = 0.0;
	// cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, devA, N, devB, K, &cublas_beta, devD, N);

	// no stream
	#ifdef COARSEN_4
	dim3 grid(M / 64, N / 64);
	smatmul_kernel_coarse4<64, 64, 8><<<grid, 256>>>(devA, devB, devC, M, N, K);
	#else
	dim3 grid(N / 128, M / 128);
	dim3 block(16, 16);
	sgemm_V2<<<grid, block>>>(devA, devB, devC, M, N, K);
	#endif // COARSEN_4

	// implicit synchronize
	CUDA_CHECK_RETURN(cudaMemcpy(C, devC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(devA));
	CUDA_CHECK_RETURN(cudaFree(devB));
	CUDA_CHECK_RETURN(cudaFree(devC));
	// CUDA_CHECK_RETURN(cudaFree(devD));
}

void smatmul_cpu_multi_threading(
	float* A, float* B, float* C, const int M, const int N, const int K
) {
	#pragma omp parallel for num_threads(OMP_THREADS)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = psum;
        }
    }
}

int main() {
	omp_set_num_threads(OMP_THREADS);
	int M = 4096, N = 2048, K = 1024;

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

	TicToc timer;

	printf("CPU SMATMUL calculating.\n");
	timer.tic();
	smatmul_cpu_multi_threading(A, B, C, M, N, K);
	float cpu_time_ms = timer.toc();
	printf("CPU SMATMUL finished in %.5f ms.\n", cpu_time_ms);
	nan_checker(C, M, N);

	printf("GPU SMATMUL calculating.\n");
	timer.tic();
	smatmul_host_caller(A, B, D, M, N, K);
	float gpu_time_ms = timer.toc();
	printf("GPU SMATMUL finished in %.5f ms.\n", gpu_time_ms);
	nan_checker(D, M, N);

	float diff = compare_result(C, D, M, N);
	printf("GPU SMATMUL: %f ms, CPU %d threads: %f ms. MAE: %.5f\n", gpu_time_ms, OMP_THREADS, cpu_time_ms, diff);

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
