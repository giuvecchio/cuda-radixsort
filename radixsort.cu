#include "cuda_runtime_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

using namespace std;

void error(const char *msg)
{
	fputs(msg, stderr);
	exit(1);
}

void check_cuda(cudaError_t err, const char *msg)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "%s - errore %d - %s\n", msg, err, cudaGetErrorString(err));
		exit(1);
	}
}

void get_vec(int numels, int *source, int *dest) {
	size_t memsize = sizeof(int)*numels;
	cudaError_t err = cudaMemcpy(dest, source, memsize,
		cudaMemcpyDeviceToHost);
	check_cuda(err, "cudaMemcpy");
}

extern __shared__ int shmem[];

__device__ int check_bit(int num, int bit_pos) {
	int bit = (num >> bit_pos) & 3;

	return bit;
}

__global__ void radixsort(int numels, const int *__restrict__ in_vec, int *__restrict__ pos_vec, int *__restrict__ state_vec, int bit_pos) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int r_number;

	if (i >= numels)
		return;

	r_number = in_vec[i];
	int check = check_bit(r_number, bit_pos);
	pos_vec[i + numels * check] = 1;
	state_vec[i] = check;
}

//SCAN
__device__ int partial_scan(const int4 *__restrict__ in, int4 * __restrict__ out,	const int offset, const int end, const int acc) {
	int4 val = make_int4(0, 0, 0, 0);
	int i = offset + threadIdx.x;
	if (i < end)
		val = in[i];
	val.y += val.x;
	val.z += val.y;
	val.w += val.z;

	__syncthreads();
	shmem[threadIdx.x] = val.w;

	int stride = 1;
	while (stride < blockDim.x) {
		/* stride threads by time works */
		int ntuple = (threadIdx.x / stride);
		/* the tread works if the n-tuple is odd */
		int work = ntuple & 1;
		/* read last value from previous n-tuple */
		int read = ntuple*stride - 1;
		__syncthreads();
		if (work)
			shmem[threadIdx.x] += shmem[read];
		stride *= 2;
	}

	__syncthreads();
	/* correction */
	int correction = acc;
	if (threadIdx.x > 0)
		correction += shmem[threadIdx.x - 1];
	val.x += correction;
	val.y += correction;
	val.z += correction;
	val.w += correction;
	if (i < end)
		out[i] = val;
	/* Update the accumulator with the tail of the last scan */
	return acc + shmem[blockDim.x - 1];
}

__global__ void scan_step1(int quart, const int4 *__restrict__ in, int4 * __restrict__ out,	int * __restrict__ aux) {
	const int quart_per_block = (quart + gridDim.x - 1) / gridDim.x;
	int offset = blockIdx.x*quart_per_block;
	/* the block works on elements with index lower than 'end' */
	int end = offset + quart_per_block;
	if (end >= quart)
		end = quart;
	/* Previous scan accumulator */
	int acc = 0;
	while (offset < end) {
		acc = partial_scan(in, out, offset, end, acc);
		offset += blockDim.x;
	}
	/* scan tail for next block */
	if (gridDim.x > 1 && threadIdx.x == blockDim.x - 1)
		aux[blockIdx.x] = acc;
}

__global__ void finalize_scan(int quart, const int *__restrict__ aux, int4 * __restrict__ scan) {
	if (blockIdx.x == 0)
		return;
	const int quart_per_block = (quart + gridDim.x - 1) / gridDim.x;
	/* beginning section index */
	int offset = blockIdx.x*quart_per_block;
	/* the block works on elements with index lower than 'end' */
	int end = offset + quart_per_block;
	if (end >= quart)
		end = quart;
	/* correction */
	int corr = aux[blockIdx.x - 1];
	int i = offset + threadIdx.x;
	while (i < end) {
		int4 val = scan[i];
		val.x += corr;
		val.y += corr;
		val.z += corr;
		val.w += corr;
		scan[i] = val;
		i += blockDim.x;
	}
}

//REORDER output
__global__ void blocks_reorder(int numels, int *__restrict__ pos_vec, int *__restrict__ state_vec, int *source, int *out_vec) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numels)
		return;
	int elem = source[i];
	out_vec[pos_vec[i + numels * state_vec[i]] - 1] = elem;
}


void radix(int numels, int *source, int *dest, int bits) {
	int blockSize = 1024;
	int numBlocks = (numels + blockSize - 1) / blockSize;
	size_t memsize = sizeof(int)*numels;

	int *step_out;
	cudaError_t err = cudaMalloc(&step_out, memsize);
	check_cuda(err, "cudaMalloc");
	err = cudaMemset(step_out, -1, memsize);
	check_cuda(err, "cudaMemset");

	int *pos_vec;
	int memsize4 = sizeof(int)*numels * 4;
	err = cudaMalloc(&pos_vec, memsize4);
	check_cuda(err, "cudaMalloc");

	int *state_vec;
	err = cudaMalloc(&state_vec, memsize);
	check_cuda(err, "cudaMalloc");

	for (int check = 0; check < bits; ) {
		err = cudaMemset(pos_vec, 0, memsize4);
		check_cuda(err, "cudaMemset");

		radixsort << <numBlocks, blockSize >> > (numels, source, pos_vec, state_vec, check);

		//POSITION SCAN
		int quart = numels;
		int aux_els = numBlocks;
		aux_els = ((aux_els + 3) / 4) * 4;
		int *d_aux = NULL;
		err = cudaMalloc(&d_aux, aux_els*sizeof(int));
		check_cuda(err, "cudaMalloc d_aux");
		scan_step1 <<<numBlocks, blockSize, sizeof(int)*blockSize >>> (quart, (int4*)pos_vec, (int4*)pos_vec, d_aux);
		scan_step1 <<<1, blockSize, sizeof(int)*blockSize >>>	(aux_els/4, (int4*)d_aux, (int4*)d_aux, NULL);
		finalize_scan <<<numBlocks, blockSize >>> (quart, d_aux, (int4*)pos_vec);

		//REORDER
		blocks_reorder << <numBlocks, blockSize >> > (numels, pos_vec, state_vec, source, step_out);
		cudaError_t err = cudaMemcpy(source, step_out, memsize, cudaMemcpyDeviceToDevice);
		check_cuda(err, "cudaMemcpy");
		check = check + 2;
	}

    get_vec(numels, source, dest);
}