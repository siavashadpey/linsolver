#ifndef CUBLAS_WRAPPER_CUH
#define CUBLAS_WRAPPER_CUH
#include <cublas_v2.h>

#include "base/cuda_header.cuh"

cublas_status cublasTnrm2(cublas_handle handle,
                          int n,
                          const float* x, int xinc, 
                          float* result)
{
    return cublasSnrm2(handle, n, x, xinc, result);
}

cublas_status cublasTnrm2(cublas_handle handle,
                          int n,
                          const double* x, int xinc, 
                          double* result)
{
    return cublasDnrm2(handle, n, x, xinc, result);
}

cublas_status cublasTdot(cublas_handle handle,
                         int n,
                         const loat* x, int xinc,
                         const loat* y, int yinc,
                         float *result)
{
    return cublasSdot(handle, n, x, xinc, result);
}

cublas_status cublasTdot(cublas_handle handle,
                         int n,
                         const double* x, int xinc,
                         const double* y, int yinc,
                         double *result)
{
    return cublasDdot(handle, n, x, xinc, result);
}

cublas_status cublasTscal(cublas_handle handle,
                          int n,
                          const float* alpha,
                          float* x, int xinc)
{
    return cublasSscal(handle, n, alpha, x, xinc);
}

cublas_status cublasTscal(cublas_handle handle,
                          int n,
                          const double* alpha,
                          double* x, int xinc)
{
    return cublasDscal(handle, n, alpha, x, xinc);
}

cublas_status cublasTaxpy(cublas_handle handle,
                          int n,
                          const float* alpha,
                          const float* x, int xinc,
                          float* y, int yinc)
{
    return cublasSaxpy(handle, n, alpha, x, xinc, y, yinc);
}

cublas_status cublasTaxpy(cublas_handle handle,
                          int n,
                          const doublet* alpha,
                          const doublet* x, int xinc,
                          double* y, int yinc)
{
    return cublasDaxpy(handle, n, alpha, x, xinc, y, yinc);
}

#endif