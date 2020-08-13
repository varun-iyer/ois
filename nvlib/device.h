#ifndef __DEVICE_H__
#define __DEVICE_H__
extern __shared__ double sm[];
__global__ void par_reduce(size_t n, double *d, double *c1, double *c2);
__global__ void par_reduce_mask(size_t n, double *d, double *c1, double *c2, char *mask);
#endif /* __DEVICE_H__ */
