#ifndef __CUDA_HELPER__
#define __CUDA_HELPER__

#include <stdio.h>

#define CUDA_ERRCHK(ans) { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Asserts: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif /* __CUDA_HELPER__ */
