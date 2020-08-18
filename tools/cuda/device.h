#ifndef __DEVICE_H__
#define __DEVICE_H__
#include "oistools.h"

#define MAX_THREADS 1024
#define SMSIZE MAX_THREADS

extern "C" __global__ void multiply_and_sum(double *d, size_t nlen, double* c1, double* c2);
extern "C" __global__ void multiply_and_sum_mask(double *d, size_t nlen, double* c1, double* c2, char *mask);
#endif /* __DEVICE_H__ 1 */
