#ifndef DEVICE_VECTOR_HELPER_CUH
#define DEVICE_VECTOR_HELPER_CUH
#include "base/cuda_header.cuh"

template <typename NumType>
__global__ void add_kernel(int size, 
               const NumType alpha,
               const NumType beta,
               const NumType* __restrict__ x,
               NumType* __restrict__ y)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= size) {
        return;
    }
    y[idx] = alpha * y[idx] + beta * x[idx];
}

template <typename NumType>
__global__ void scale_add_kernel(int size, 
                     const NumType alpha,
                     const NumType* __restrict__ x,
                     NumType* __restrict__ y)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= size) {
        return;
    }
    y[idx] = alpha * y[idx] + x[idx];
}

template <typename NumType>
__global__ void elementwise_multiply_kernel(int size, 
                     const NumType* __restrict__ x,
                     NumType* __restrict__ y)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= size) {
        return;
    }
    y[idx] = y[idx] * x[idx];
}

template <typename NumType>
__global__ void elementwise_multiply_kernel(int size, 
                     const NumType* __restrict__ a,
                     const NumType* __restrict__ b,
                     NumType* __restrict__ c)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= size) {
        return;
    }
    c[idx] = a[idx] * b[idx];
}

#endif