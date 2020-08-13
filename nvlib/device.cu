#include "device.h"

__global__ void par_reduce(double *d, double *c1, double *c2) {
	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[tid] = c1[i] * c2[i]; // copy to SM
	for (unsigned int s = blockDim.x / 2; s > 0; s>>=1) {
		if(tid < s) {
			sm[tid] += sm[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) d[blockIdx.x] = sm[0];
	// d[blockIdx.x] is the sum of the block
}

__global__ void par_reduce_mask(double *d, double *c1, double *c2, char *mask) {
	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[tid] = mask[i] ? 0 : c1[i] * c2[i]; // copy to SM
	for (unsigned int s = blockDim.x / 2; s > 0; s>>=1) {
		if(tid < s) {
			sm[tid] += sm[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) d[blockIdx.x] = sm[0];
}
