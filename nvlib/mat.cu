#define CUDA
#ifndef CUDA

double multiply_and_sum(size_t nsize, double* C1, double* C2) {
    double result = 0.0;
    for (size_t i = 0; i < nsize; i++) {
        result += C1[i] * C2[i];
    }
    return result;
}


double multiply_and_sum_mask(size_t nsize, double* C1, double* C2, char* mask) {
    double result = 0.0;
    for (size_t i = 0; i < nsize; i++) {
        if (mask[i] == 0) result += C1[i] * C2[i];
    }
    return result;
}

#endif /* CUDA */
