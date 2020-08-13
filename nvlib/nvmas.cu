#include <stdio.h>
#include "device.h"
#include "nvmas.h"

#define MAX_THREADS 1024
#define SMSIZE (sizeof(double) * MAX_THREADS)

extern "C" double multiply_and_sum(size_t nsize, double* h_C1, double* h_C2) {
	double *d_C1, *d_C2, *d;

	cudaMalloc(&d_C1, nsize * sizeof(double));
	cudaMalloc(&d_C2, nsize * sizeof(double));
	cudaMalloc(&d, nsize * sizeof(double));

	cudaMemcpy(d_C1, h_C1, nsize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2, h_C2, nsize * sizeof(double), cudaMemcpyHostToDevice);

	dim3 dimBlock(MAX_THREADS, 1, 1);
	dim3 dimGrid(nsize / MAX_THREADS + 1, 1, 1);
	par_reduce<<<dimGrid, dimBlock, SMSIZE>>>(nsize, d, d_C1, d_C2);

	double h_r;
	cudaMemcpy(&h_r, d, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_C1);
	cudaFree(d_C2);
	cudaFree(d);

    return h_r;
}

extern "C" double multiply_and_sum_mask(size_t nsize, double* h_C1, double* h_C2, char *h_m) {
	double *d_C1, *d_C2, *d;
	char *d_m;

	cudaMalloc(&d_C1, nsize * sizeof(double));
	cudaMalloc(&d_C2, nsize * sizeof(double));
	cudaMalloc(&d_m, nsize * sizeof(char));
	cudaMalloc(&d, nsize * sizeof(double));

	cudaMemcpy(d_C1, h_C1, nsize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2, h_C2, nsize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, h_m, nsize * sizeof(char), cudaMemcpyHostToDevice);

	dim3 dimBlock(MAX_THREADS, 1, 1);
	dim3 dimGrid(nsize / MAX_THREADS + 1, 1, 1);
	par_reduce_mask<<<dimGrid, dimBlock, SMSIZE>>>(nsize, d, d_C1, d_C2, d_m);

	double h_r;
	cudaMemcpy(&h_r, d, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_C1);
	cudaFree(d_C2);
	cudaFree(d);
    return h_r;
}
