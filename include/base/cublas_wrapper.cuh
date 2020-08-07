#ifndef CUBLAS_WRAPPER_CUH
#define CUBLAS_WRAPPER_CUH
#include <cublas_v2.h>

#include "base/cuda_header.cuh"

inline cublasStatus_t cublasTnrm2(cublasHandle_t handle,
                          int n,
                          const float* x, int xinc, 
                          float* result)
{
    return cublasSnrm2(handle, n, x, xinc, result);
}

inline cublasStatus_t cublasTnrm2(cublasHandle_t handle,
                          int n,
                          const double* x, int xinc, 
                          double* result)
{
    return cublasDnrm2(handle, n, x, xinc, result);
}

inline cublasStatus_t cublasTdot(cublasHandle_t handle,
                         int n,
                         const float* x, int xinc,
                         const float* y, int yinc,
                         float *result)
{
    return cublasSdot(handle, n, x, xinc, y, yinc, result);
}

inline cublasStatus_t cublasTdot(cublasHandle_t handle,
                         int n,
                         const double* x, int xinc,
                         const double* y, int yinc,
                         double *result)
{
    return cublasDdot(handle, n, x, xinc, y, yinc, result);
}

inline cublasStatus_t cublasTscal(cublasHandle_t handle,
                          int n,
                          const float* alpha,
                          float* x, int xinc)
{
    return cublasSscal(handle, n, alpha, x, xinc);
}

inline cublasStatus_t cublasTscal(cublasHandle_t handle,
                          int n,
                          const double* alpha,
                          double* x, int xinc)
{
    return cublasDscal(handle, n, alpha, x, xinc);
}

inline cublasStatus_t cublasTaxpy(cublasHandle_t handle,
                          int n,
                          const float* alpha,
                          const float* x, int xinc,
                          float* y, int yinc)
{
    return cublasSaxpy(handle, n, alpha, x, xinc, y, yinc);
}

inline cublasStatus_t cublasTaxpy(cublasHandle_t handle,
                          int n,
                          const double* alpha,
                          const double* x, int xinc,
                          double* y, int yinc)
{
    return cublasDaxpy(handle, n, alpha, x, xinc, y, yinc);
}

#endif