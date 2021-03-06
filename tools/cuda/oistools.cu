#include "oistools.h"
#include "cuda_helper.h"

#define MAX_THREADS 1024
#define SMSIZE (MAX_THREADS * sizeof(double))

void fill_c_matrices_for_kernel(int k_height, int k_width, int deg, int n, int m, double* refimage, double* Conv);
void fill_c_matrices_for_background(int n, int m, int bkg_deg, double* Conv_bkg);
extern "C" __global__ void multiply_and_sum(double *d, size_t nlen, double* c1, double* c2);
extern "C" __global__ void multiply_and_sum_mask(double *d, size_t nlen, double* c1, double* c2, char *mask);

extern "C" lin_system build_matrix_system(int n, int m, double* image, double* refimage,
    int kernel_height, int kernel_width, int kernel_polydeg, int bkg_deg,
    char *mask)
{
    int kernel_len = kernel_height * kernel_width;
    int img_len = n * m;
    int kpdeg = kernel_polydeg;
    int poly_degree = (kpdeg + 1) * (kpdeg + 2) / 2;
    int bkg_dof;

    bkg_dof = (bkg_deg + 1) * (bkg_deg + 2) / 2;

    size_t conv_len = ((size_t) img_len) * (kernel_len * poly_degree + bkg_dof);
    double* Conv = (double *) calloc(conv_len, sizeof(*Conv)); // TODO err on bad calloc

    fill_c_matrices_for_kernel(kernel_height, kernel_width, kernel_polydeg, n, m, refimage, Conv);
    double* Conv_bkg;
    if (bkg_deg != -1) {
        Conv_bkg = Conv + img_len * kernel_len * poly_degree;
        fill_c_matrices_for_background(n, m, bkg_deg, Conv_bkg);
    }

    int total_dof = kernel_len * poly_degree + bkg_dof;
	size_t M_size = ((size_t) total_dof) * total_dof * sizeof(double);
	size_t b_size = ((size_t) total_dof) * sizeof(double);

	double *d_M, *d_b, *d_Conv, *d_img;
	char *d_m;
	CUDA_ERRCHK(cudaMalloc(&d_M, M_size));
	CUDA_ERRCHK(cudaMalloc(&d_b, b_size));
	CUDA_ERRCHK(cudaMalloc(&d_Conv, conv_len * sizeof(double)));
	CUDA_ERRCHK(cudaMalloc(&d_img, img_len * sizeof(double)));
	if(mask) {
		CUDA_ERRCHK(cudaMalloc(&d_m, conv_len * sizeof(char)));
		CUDA_ERRCHK(cudaMemcpy(d_m, mask, img_len * sizeof(double), cudaMemcpyHostToDevice));
	}
	CUDA_ERRCHK(cudaMemcpy(d_Conv, Conv, conv_len * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_ERRCHK(cudaMemcpy(d_img, image, img_len * sizeof(double), cudaMemcpyHostToDevice));

	dim3 dimBlock(MAX_THREADS, 1, 1);
	dim3 dimGrid(img_len / MAX_THREADS + 1, 1, 1);

	for(size_t i = 0; i < total_dof; i++) {
		double* d_c1 = d_Conv + i * img_len;
		for(size_t j = i; j < total_dof; j++) {
			double *d_c2 = d_Conv + j * img_len;
			if(mask) multiply_and_sum_mask<<<dimGrid, dimBlock, SMSIZE>>>(d_M + i * total_dof + j, img_len, d_c1, d_c2, d_m);
			else multiply_and_sum<<<dimGrid, dimBlock, SMSIZE>>>(d_M + i * total_dof + j, img_len, d_c1, d_c2);
			cudaMemcpy(d_M + j * total_dof + i, d_M + i * total_dof + j, sizeof(double), cudaMemcpyDeviceToDevice);
		}
		if(mask) multiply_and_sum_mask<<<dimGrid, dimBlock, SMSIZE>>>(d_b + i, img_len, image, d_c1, d_m);
		else multiply_and_sum<<<dimGrid, dimBlock, SMSIZE>>>(d_b + i, img_len, d_img, d_c1);
	}

	double *M = (double *) malloc(M_size);
	double *b = (double *) malloc(b_size);
	CUDA_ERRCHK(cudaMemcpy(M, d_M, M_size, cudaMemcpyDeviceToHost));
	CUDA_ERRCHK(cudaMemcpy(b, d_b, b_size, cudaMemcpyDeviceToHost));
	CUDA_ERRCHK(cudaFree(d_M));
	CUDA_ERRCHK(cudaFree(d_b));
	CUDA_ERRCHK(cudaFree(d_Conv));
	CUDA_ERRCHK(cudaFree(d_img));
	free(Conv);
	if(mask) CUDA_ERRCHK(cudaFree(d_m));

    lin_system the_system = {total_dof, M, b};
    return the_system;
}


extern "C" void convolve2d_adaptive(int n, int m, double* image,
		 
                            int kernel_height, int kernel_width,
                            int kernel_polydeg, double* kernel,
                            double* Conv)
{
    //int k_side = kernel_height;
    int k_poly_dof = (kernel_polydeg + 1) * (kernel_polydeg + 2) / 2;

    for (long conv_row = 0; conv_row < n; ++conv_row) {
        for (long conv_col = 0; conv_col < m; ++conv_col) {
            int conv_index = conv_row * m + conv_col;

            for (int p = 0; p < kernel_height; p++) {
                for (int q = 0; q < kernel_width; q++) {
                    long img_row = conv_row - (p - kernel_height / 2); // khs is kernel half side
                    long img_col = conv_col - (q - kernel_width / 2);
                    size_t img_index = img_row * m + img_col;

                    // do only if img_index is in bounds of image
                    if (img_row >= 0 && img_col >=0 && img_row < n && img_col < m) {

                        // reconstruct the (p, q) pixel of kernel
                        double k_pixel = 0.0;
                        // advance k_coeffs pointer to the p, q part
                        double* k_coeffs_pq = kernel + (p * kernel_width + q) * k_poly_dof;
                        size_t exp_index = 0;
                        for (int exp_x = 0; exp_x <= kernel_polydeg; exp_x++) {
                            for (int exp_y = 0; exp_y <= kernel_polydeg - exp_x; exp_y++) {
                                k_pixel += k_coeffs_pq[exp_index] * pow(conv_row, exp_y) * pow(conv_col, exp_x);
                                exp_index++;
                            }
                        }

                        Conv[conv_index] += image[img_index] * k_pixel;
                    }
                }
            }

        } // conv_col
    } // conv_row

}

void fill_c_matrices_for_kernel(int k_height, int k_width, int deg, int n, int m, double* refimage, double* Conv) {

    size_t img_size = n * m;
    int poly_degree = (deg + 1) * (deg + 2) / 2;

    for (size_t p = 0; p < k_height; p++) {
        for (size_t q = 0; q < k_width; q++) {
            double* Conv_pq = Conv + (p * k_width + q) * poly_degree * img_size;

            size_t exp_index = 0;
            for (int exp_x = 0; exp_x <= deg; exp_x++) {
                for (int exp_y = 0; exp_y <= deg - exp_x; exp_y++) {
                    double* Conv_pqkl = Conv_pq + exp_index * img_size;

                    for (long conv_row = 0; conv_row < n; ++conv_row) {
                        for (long conv_col = 0; conv_col < m; ++conv_col) {
                            size_t conv_index = conv_row * m + conv_col;
                            long img_row = conv_row - (p - k_height / 2); // khs is kernel half side
                            long img_col = conv_col - (q - k_width / 2);
                            size_t img_index = img_row * m + img_col;
                            double x_pow = pow(conv_col, exp_x);
                            double y_pow = pow(conv_row, exp_y);
                            // make sure img_index is in bounds of refimage
                            if (img_row >= 0 && img_col >=0 && img_row < n && img_col < m) {
                                Conv_pqkl[conv_index] = refimage[img_index] * x_pow * y_pow;
                            }
                        } // conv_col
                    } // conv_row

                    exp_index++;
                } // exp_y
            } // exp_x

        } //q
    } // p

    return;
}

void fill_c_matrices_for_background(int n, int m, int bkg_deg, double* Conv_bkg) {

    int exp_index = 0;
    for (size_t exp_x = 0; exp_x <= bkg_deg; exp_x++) {
        for (size_t exp_y = 0; exp_y <= bkg_deg - exp_x; exp_y++) {

            double* Conv_xy = Conv_bkg + exp_index * n * m;

            for (long conv_row = 0; conv_row < n; ++conv_row) {
                for (long conv_col = 0; conv_col < m; ++conv_col) {
                    size_t conv_index = conv_row * m + conv_col;
                    double x_pow = pow(conv_col, exp_x);
                    double y_pow = pow(conv_row, exp_y);

                    Conv_xy[conv_index] = x_pow * y_pow;
                } // conv_col
            } // conv_row

            exp_index++;
        } // exp_y
    } // exp_x

    return;
}

extern "C" __global__ void multiply_and_sum(double *d, size_t nlen, double *c1, double *c2) {
	extern __shared__ double sm[];
	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[tid] = (i < nlen) ? c1[i] * c2[i] : 0; // copy to SM
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
