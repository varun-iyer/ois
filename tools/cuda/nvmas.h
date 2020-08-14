#ifndef __LNVMAS__
#define __LNVMAS__
extern "C" double multiply_and_sum(size_t nsize, double* h_C1, double* h_C2);
extern "C" double multiply_and_sum_mask(size_t nsize, double* h_C1, double* h_C2, char *h_m);
#endif /* __LNVMAS__ */
