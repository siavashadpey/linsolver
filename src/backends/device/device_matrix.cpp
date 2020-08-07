#include <stdio.h>
#include <cmath>
#include <cassert>

#include "base/cuda_header.cuh"
#include "base/cublas_wrapper.cuh"
#include "backends/device/device_vector_helper.cuh"
#include "base/error.h"

#include "backends/device/device_matrix.h"
#include "backends/device/device_vector.h"

template <typename NumType>
DeviceMatrix<NumType>::DeviceMatrix()
    :   val_(nullptr),
        row_ptr_(nullptr),
        col_idx_(nullptr)
{
    CUBLAS_CALL( cublasCreate(&cublasHandle_) );
    CUSPARSE_CALL( cusparseCreate(&cusparseHandle_) );
}

template <typename NumType>
DeviceMatrix<NumType>::~DeviceMatrix()
{
    CUBLAS_CALL( cublasDestroy(cublasHandle_) );
    CUSPARSE_CALL( cusparseDestroy(cusparseHandle_) );
    this->clear();
}

template <typename NumType>
void DeviceMatrix<NumType>::allocate(int m, int n, int nnz)
{
    assert(nnz > 0);
    this->clear();

    this->m_ = m;
    this->n_ = n;
    this->nnz_ = nnz;

    CUDA_CALL( cudaMalloc( (void**) &(this->val_), nnz * sizeof(NumType)) );
    CUDA_CALL( cudaMalloc( (void**) &(this->row_ptr_), (m + 1) * sizeof(int)) );
    CUDA_CALL( cudaMalloc( (void**) &(this->col_idx_), nnz * sizeof(int)) );
    this->col_idx_ = new int[nnz];
}

template <typename NumType>
void DeviceMatrix<NumType>::clear()
{
    if (this-> nnz_ > 0) {
        CUDA_CALL( cudaFree(this->val_) );
        CUDA_CALL( cudaFree(this->row_ptr_) );
        CUDA_CALL( cudaFree(this->col_idx_) );
        this->m_ = 0;
        this->n_ = 0;
        this->nnz_ = 0;
    }
}

template <typename NumType>
void DeviceMatrix<NumType>::copy_from(const NumType* val, const int* row_ptr, const int* col_idx)
{
    Error("Method has not yet been implemented.");
}

template <typename NumType>
void DeviceMatrix<NumType>::copy_from(const BaseMatrix<NumType>& B)
{
    Error("Method has not yet been implemented.");
}

template <typename NumType>
NumType DeviceMatrix<NumType>::norm() const
{
    NumType val = static_cast<NumType>(0);

    assert(this->nnz_ > 0);

    CUBLAS_CALL( cublasTnrm2(cublasHandle_, 
                             this->nnz_, 
                             this->val_, 
                             1,             // increment of 1
                             &val));

    return val;
}

template <typename NumType>
void DeviceMatrix<NumType>::scale(NumType alpha)
{
    assert(this->nnz_ > 0);
    CUBLAS_CALL( cublasTscal(cublasHandle_,
                              this->nnz_, 
                              &alpha,
                              this->val_, 1 // increment of 1
                              ));
}


template <typename NumType>
void DeviceMatrix<NumType>::multiply(const BaseVector<NumType>& v_in, 
        BaseVector<NumType>* w_out) const
{
    Error("Method has not yet been implemented.");
}

template <typename NumType>
bool DeviceMatrix<NumType>::read_matrix_market(const std::string filename) {
    Error("Method has not yet been implemented.");
}

// instantiate template classes
template class DeviceMatrix<double>;
template class DeviceMatrix<float>;
