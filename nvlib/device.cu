#include "device.h"

__global__ void par_reduce(double *d, double *c1, double *c2) {
	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[tid] = c1[i] * c2[i]; // copy to SM
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if (tid % (2 * stride) == 0) sm[tid] += sm[tid + stride];
	}
	if (tid == 0) d[blockIdx.x] = sm[0];
	// d[blockIdx.x] is the sum of the block
}
