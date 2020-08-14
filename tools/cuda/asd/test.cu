#include "util.h"

__global__ void my_dbl_sq(float *val){
  *val = 2*my_square(*val);
}

float dbl_sq(float val){
  float *d_val, h_val;
  cudaMalloc(&d_val, sizeof(float));
  h_val = val;
  cudaMemcpy(d_val, &h_val, sizeof(float), cudaMemcpyHostToDevice);
  my_dbl_sq<<<1,1>>>(d_val);
  cudaMemcpy(&h_val, d_val, sizeof(float), cudaMemcpyDeviceToHost);
  return h_val;
}
