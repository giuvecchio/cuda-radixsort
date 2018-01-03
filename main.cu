#include "cuda_runtime_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include "radixsort.cu"

#define ord_gen 0 //Set to 1 to order fill the array, 0 to reverse fill

using namespace std;

void error(const char *msg){
	fputs(msg, stderr);
	exit(1);
}

void check_cuda(cudaError_t err, const char *msg){
	if (err != cudaSuccess) {
		fprintf(stderr, "%s - errore %d - %s\n", msg, err, cudaGetErrorString(err));
		exit(1);
	}
}

//ARRAY FILL
__global__ void ord_init(int numels, int *vec, int offset) { //Ordered array fill
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	i += offset;
	if (i >= numels)
		return;
	vec[i] = i;
}

__global__ void rev_init(int numels, int *vec, int offset) { //Reverse array fill
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	i += offset;
	if (i >= numels)
		return;
	vec[i] = numels - i - 1;
}

void vec_init(int numels, int *source) {
	cudaError_t err;

	cudaDeviceProp props;
	err = cudaGetDeviceProperties(&props, 0);
	check_cuda(err, "cudaGetDeviceProperties");
	const int maxBlocks = props.maxGridSize[0];
	int blockSize = props.maxThreadsDim[0];
	int totalBlocks = (numels + blockSize - 1) / blockSize;
#if ord_gen
	int elementsDone = 0;
	while (totalBlocks > maxBlocks) {
		ord_init << <maxBlocks, blockSize >> >
			(numels, source, elementsDone);
		elementsDone += maxBlocks*blockSize;
		totalBlocks -= maxBlocks;
	}
	ord_init << <totalBlocks, blockSize >> >
		(numels, source, elementsDone);
#else
	int elementsDone = 0;
	while (totalBlocks > maxBlocks) {
		rev_init << <maxBlocks, blockSize >> >
			(numels, source, elementsDone);
		elementsDone += maxBlocks*blockSize;
		totalBlocks -= maxBlocks;
	}
	rev_init << <totalBlocks, blockSize >> >
		(numels, source, elementsDone);
#endif
}

int main(int argc, char *argv[]) {

	if (argc <= 2)
		error("Chore number of elements and bits to order\n");

	int numels = atoi(argv[1]);
	int bits = atoi(argv[2]); //Number of the most significant bits to be ordered
	
    int *source, *dest;

    init(numels, source);

	radix(numels, source, dest, bits);
    /* The output of the sort will be stored into the dest vec*/
	
	printf("Sorting of %d elements successfully completed.\n", numels);

	cudaFree(source);
	free(dest);

	err = cudaDeviceReset();
	check_cuda(err, "cudaDeviceReset");

	return 0;
}

void init(int numels, int *source){
    size_t memsize = sizeof(int)*numels;

	dest = (int*)malloc(memsize);
	if (!dest)
		error("malloc");

	cudaError_t err = cudaMalloc(&source, memsize);
	check_cuda(err, "cudaMalloc");
	err = cudaMemset(source, -1, memsize);
	check_cuda(err, "cudaMemset");

	vec_init(numels, source);
}