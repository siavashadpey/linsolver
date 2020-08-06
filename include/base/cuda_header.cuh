#ifndef CUDA_HEADER_CUH
#define CUDA_HEADER_CUH

#ifdef __CUDACC__

// Device function attributes
#include <cuda_runtime.h>
#include "cuda_error_helper.cuh"

#define CUDA_CALLABLE __host__ __device__

#else

// Host function attributes
#define CUDA_CALLABLE

#endif // __CUDA_ARCH__

#endif // CUDA_HEADER_CUH_
