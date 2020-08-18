#include "device.h"

extern "C" __global__ void multiply_and_sum(double *d, size_t nlen, double *c1, double *c2) {
	extern __shared__ double sm[];
	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[tid] = i<nlen ? c1[i] * c2[i] : 0; // copy to SM
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s>>=1) {
		if(tid < s) {
			sm[tid] += sm[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) atomicAdd(d, sm[0]);
	// d[blockIdx.x] is the sum of the block
}

extern "C" __global__ void multiply_and_sum_mask(double *d, size_t nlen, double *c1, double *c2, char *mask) {
	extern __shared__ double sm[];
	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[tid] = (mask[i] || i >= nlen) ? 0 : c1[i] * c2[i]; // copy to SM
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s>>=1) {
		if(tid < s) {
			sm[tid] += sm[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) atomicAdd(d, sm[0]);
}
