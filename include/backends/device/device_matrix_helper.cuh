#ifndef DEVICE_MATRIX_HELPER_CUH
#define DEVICE_MATRIX_HELPER_CUH
#include "base/cuda_header.cuh"


template <typename NumType>
__global__ void get_diag_kernel(int row_size, 
                     const int* __restrict__ row_ptr,
                     const int* __restrict__ col_idx,
                     const NumType* __restrict__ A,
                     NumType* __restrict__ diag)
{
    int irow = threadIdx.x + blockDim.x*blockIdx.x;
    if (irow >= row_size) {
        return;
    }
    
    for (int val_idx = row_ptr[irow]; val_idx < row_ptr[irow+1]; val_idx++) {
        int jcol = col_idx[val_idx];
        if (irow == jcol) {
            diag[irow] = A[val_idx];
            break;
        }
    }
}

template <typename NumType>
__global__ void compute_inverse_diag_kernel(int row_size, 
                     const int* __restrict__ row_ptr,
                     const int* __restrict__ col_idx,
                     const NumType* __restrict__ A,
                     NumType* __restrict__ inv_diag)
{
    const NumType one = static_cast<NumType>(1);

    int irow = threadIdx.x + blockDim.x*blockIdx.x;
    if (irow >= row_size) {
        return;
    }
    
    for (int val_idx = row_ptr[irow]; val_idx < row_ptr[irow+1]; val_idx++) {
        int jcol = col_idx[val_idx];
        if (irow == jcol) {
            inv_diag[irow] = one/A[val_idx];
            break;
        }
    }
}

#endif