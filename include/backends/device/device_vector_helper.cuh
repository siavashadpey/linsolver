#ifndef DEVICE_VECTOR_HELPER_CUH
#define DEVICE_VECTOR_HELPER_CUH
#include "base/cuda_header.cuh"

template <typename NumType>
__global__ add(int size, 
               const NumType* __restrict__ alpha,
               const NumType* __restrict__ beta ,
               const NumType* __restrict__ x    ,
               NumType* __restrict__ y)
{
    int idx = ThreadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= n) {
        return;
    }
    y[idx] = (*alpha) * y[idx] + (*beta) * x[idx];
}

template <typename NumType>
__global__ scale_add(int size, 
                     const NumType* __restrict__ alpha,
                     const NumType* __restrict__ x    ,
                     NumType* __restrict__ y)
{
    int idx = ThreadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= n) {
        return;
    }
    y[idx] = (*alpha) * y[idx] + x[idx];
}

#endif