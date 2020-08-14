#include <stdio.h>
#include "device.h"
#include "nvmas.h"

#define MAX_THREADS 1024
#define SMSIZE (sizeof(double) * MAX_THREADS)

extern "C" double multiply_and_sum(size_t nlen, double* h_C1, double* h_C2) {
	double *d_C1, *d_C2, *d;

	cudaMalloc(&d_C1, nlen * sizeof(double));
	cudaMalloc(&d_C2, nlen * sizeof(double));
	cudaMalloc(&d, nlen * sizeof(double));

	cudaMemcpy(d_C1, h_C1, nlen * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2, h_C2, nlen * sizeof(double), cudaMemcpyHostToDevice);

	dim3 dimBlock(MAX_THREADS, 1, 1);
	dim3 dimGrid(nlen / MAX_THREADS + 1, 1, 1);
	par_reduce<<<dimGrid, dimBlock, SMSIZE>>>(nlen, d, d_C1, d_C2);

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

extern "C" double d_convolve(double *image, size_t image_len, double *Conv, double *M, double *b, size_t conv_size, int total_dof, char *mask) {
	double *d_M, *d_b *d_Conv, *d_m, *d_img, *d_out;
	cudaMalloc(&d_M, M_size);
	cudaMalloc(&d_b, b_size);
	cudaMalloc(&d_Conv, conv_size);
	cudaMalloc(&d_img, image_len * sizeof(double));
	cudaMalloc(&d_out, sizeof(double));
	if(mask) {
		cudaMalloc(&d_m, conv_size);
		cudaMemcpy(d_m, mask, img_len * sizeof(double), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_Conv, Conv, conv_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_img, Image, conv_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(MAX_THREADS, 1, 1);
	dim3 dimGrid(nsize / MAX_THREADS + 1, 1, 1);

	for(size_t i = 0; i < total_dof; i++) {
		double* d_c1 = d_Conv + i * img_len;
		for(size_t j = i; j < total_dof; j++) {
			double *d_c2 = d_Conv + j * img_len;
			if(mask) par_reduce_mask<<<dimGrid, dimBlock, SMSIZE>>>(img_len, d_M + i * total_dof + j, d_c1, d_c2, d_m);
			else par_reduce<<<dimGrid, dimBlock, SMSIZE>>>(img_len, d_M + i * total_dof + j, d_c1, d_c2, d_m);
			d_M[j * total_dof + i] = d_M[i * total_dof + j];
		}
		if(mask) par_reduce<<<dimGrid, dimBlock, SMSIZE>>>(img_len, b + i, image, d_c1);
		else par_reduce<<<dimGrid, dimBlock, SMSIZE>>>(img_len, d_b + i, image, d_c1, d_m);
	}
}
