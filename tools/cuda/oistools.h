#ifndef __OISTOOLS_H__
#define __OISTOOLS_H__

#include <stdlib.h>
#include <math.h>

typedef struct {
    int b_dim;
    double* M;
    double* b;
} lin_system;

extern "C" lin_system build_matrix_system(int n, int m, double* image, double* refimage,
                         int kernel_height, int kernel_width, int kernel_polydeg,
                         int bkg_deg, char *mask);

extern "C" void convolve2d_adaptive(int n, int m, double* image,
                            int kernel_height, int kernel_width, int kernel_polydeg,
                            double* kernel, double* convolution);
#endif /* __OISTOOLS_H__ 1 */
