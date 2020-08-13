#include "device.h"
#include "nvmas.h"

extern "C" double multiply_and_sum(size_t nsize, double* h_C1, double* h_C2) {
	size_t size = nsize * sizeof(double);

	int threads = nsize;
	int blocks = nsize / threads; // number of blocks
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smSize = threads * sizeof(int);

	double *d_C1, *d_C2, *d;

	cudaMalloc(&d_C1, nsize * sizeof(double));
	cudaMalloc(&d_C2, nsize * sizeof(double));
	cudaMalloc(&d, nsize * sizeof(double));

	cudaMemcpy(d_C1, h_C1, nsize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2, h_C2, nsize * sizeof(double), cudaMemcpyHostToDevice);

	par_reduce<<<dimGrid, dimBlock, smSize>>>(d, d_C1, d_C2);

	uint todo;
	if (blocks > 1) todo = 1 + blocks/128;
	else todo = 0;

	for(int i = 0; i < todo; i++) {
		threads = (blocks < nsize) ? blocks : nsize;
		blocks = blocks / threads;
		dim3 dimBlock(threads, 1, 1);
		dim3 dimGrid(blocks, 1, 1);
		par_reduce<<<dimGrid, dimBlock, smSize>>>(d, d_C1, d_C2); 
	}

	double h_r;
	cudaMemcpy(&h_r, d, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_C1);
	cudaFree(d_C2);
    return h_r;
}

extern "C" double multiply_and_sum_mask(size_t nsize, double* h_C1, double* h_C2, char *h_m) {
	size_t size = nsize * sizeof(double);

	int threads = nsize;
	int blocks = nsize / threads; // number of blocks
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smSize = threads * sizeof(int);

	double *d_C1, *d_C2, *d;
	char *d_m;

	cudaMalloc(&d_C1, nsize * sizeof(double));
	cudaMalloc(&d_C2, nsize * sizeof(double));
	cudaMalloc(&d_m, nsize * sizeof(char));
	cudaMalloc(&d, nsize * sizeof(double));

	cudaMemcpy(d_C1, h_C1, nsize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2, h_C2, nsize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, h_m, nsize * sizeof(char), cudaMemcpyHostToDevice);

	par_reduce_mask<<<dimGrid, dimBlock, smSize>>>(d, d_C1, d_C2, d_m);

	uint todo;
	if (blocks > 1) todo = 1 + blocks/128;
	else todo = 0;

	for(int i = 0; i < todo; i++) {
		threads = (blocks < nsize) ? blocks : nsize;
		blocks = blocks / threads;
		dim3 dimBlock(threads, 1, 1);
		dim3 dimGrid(blocks, 1, 1);
		par_reduce_mask<<<dimGrid, dimBlock, smSize>>>(d, d_C1, d_C2, d_m); 
	}

	double h_r;
	cudaMemcpy(&h_r, d, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_C1);
	cudaFree(d_C2);
    return h_r;
}
