#include <stdio.h>
#include <cmath>
#include <cassert>

#include "base/cuda_header.cuh"
#include "base/backend_wrapper.h"
#include "base/cublas_wrapper.cuh"
#include "base/cusparse_wrapper.cuh"
#include "backends/device/device_vector_helper.cuh"
#include "backends/device/device_matrix_helper.cuh"
#include "base/error.h"

#include "backends/device/device_matrix.h"
#include "backends/device/device_vector.h"
#include "backends/host/host_matrix.h"

template <typename NumType>
DeviceMatrix<NumType>::DeviceMatrix()
    :   val_(nullptr),
        row_ptr_(nullptr),
        col_idx_(nullptr)
{
    CUSPARSE_CALL( cusparseCreateMatDescr(&cusparseMatDescr_));
    CUSPARSE_CALL( cusparseSetMatIndexBase(cusparseMatDescr_, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CALL( cusparseSetMatType(cusparseMatDescr_, CUSPARSE_MATRIX_TYPE_GENERAL));
}

template <typename NumType>
DeviceMatrix<NumType>::~DeviceMatrix()
{
    CUSPARSE_CALL( cusparseDestroyMatDescr(cusparseMatDescr_));
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

    const NumType zero = static_cast<NumType>(0);
    CUDA_CALL(cudaMemset(this->val_, zero, nnz * sizeof(NumType)));
    CUDA_CALL(cudaMemset(this->row_ptr_, zero, (m + 1) * sizeof(int)));
    CUDA_CALL(cudaMemset(this->col_idx_, zero, nnz * sizeof(int)));
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
    assert(this->m_ > 0);
    assert(this->n_ > 0);
    assert(this->nnz_ > 0);

    CUDA_CALL( cudaMemcpy( this->val_, val, this->nnz_ * sizeof(NumType), cudaMemcpyDeviceToDevice));
    CUDA_CALL( cudaMemcpy( this->row_ptr_, row_ptr, (this->m_ + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CALL( cudaMemcpy( this->col_idx_, col_idx, this->nnz_ * sizeof(int), cudaMemcpyDeviceToDevice));   
}

template <typename NumType>
void DeviceMatrix<NumType>::copy_from(const BaseMatrix<NumType>& B)
{
    const DeviceMatrix<NumType>* B_device = dynamic_cast<const DeviceMatrix<NumType>*>(&B);
    const HostMatrix<NumType>* B_host = dynamic_cast<const HostMatrix<NumType>*>(&B);

    // inputted matrix is on device
    if (B_device != nullptr) {
        if ((B_device->nnz_ != this->nnz_) or 
        (B_device->m_ != this->m_) or
        (B_device->n_ != this->n_)) {
            this->allocate(B_device->m_, B_device->n_, B_device->nnz_);
        }

        CUDA_CALL( cudaMemcpy( this->val_, B_device->val_, this->nnz_ * sizeof(NumType), cudaMemcpyDeviceToDevice));
        CUDA_CALL( cudaMemcpy( this->row_ptr_, B_device->row_ptr_, (this->m_ + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CALL( cudaMemcpy( this->col_idx_, B_device->col_idx_, this->nnz_ * sizeof(int), cudaMemcpyDeviceToDevice));   
    }
    // inputted vector is on host
    else if (B_host != nullptr) {
        if ((B_host->nnz_ != this->nnz_) or 
        (B_host->m_ != this->m_) or
        (B_host->n_ != this->n_)) {
            this->allocate(B_host->m_, B_host->n_, B_host->nnz_);
        }

        CUDA_CALL( cudaMemcpy( this->val_, B_host->val_, this->nnz_ * sizeof(NumType), cudaMemcpyHostToDevice));
        CUDA_CALL( cudaMemcpy( this->row_ptr_, B_host->row_ptr_, (this->m_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CALL( cudaMemcpy( this->col_idx_, B_host->col_idx_, this->nnz_ * sizeof(int), cudaMemcpyHostToDevice));   
    }
    else {
        Error("Cannot cast matrix as DeviceMatrix or HostMatrix.");
    }
}

template <typename NumType>
void DeviceMatrix<NumType>::copy_to(BaseMatrix<NumType>& ) const {
    Error("Method has not yet been implemented.");
}

template <typename NumType>
NumType DeviceMatrix<NumType>::norm() const
{
    NumType val = static_cast<NumType>(0);

    assert(this->nnz_ > 0);

    CUBLAS_CALL( cublasTnrm2(manager::get_backend_struct().cublasHandle, 
                             this->nnz_, 
                             this->val_, 
                             1,             // increment of 1
                             &val));

    return val;
}

template <typename NumType>
void DeviceMatrix<NumType>::scale(NumType alpha)
{
    const NumType one = static_cast<NumType>(1);

    if (alpha == one){
        return;
    }
    
    assert(this->nnz_ > 0);
    CUBLAS_CALL( cublasTscal(manager::get_backend_struct().cublasHandle,
                              this->nnz_, 
                              &alpha,
                              this->val_, 1 // increment of 1
                              ));
}


template <typename NumType>
void DeviceMatrix<NumType>::multiply(const BaseVector<NumType>& v_in, 
        BaseVector<NumType>* w_out) const
{
    assert(this->n_ > 0);
    assert(this->m_ > 0);
    assert(v_in.n() == this->n_);
    assert(w_out->n() == this->m_);

    const DeviceVector<NumType>* v_in_d = dynamic_cast<const DeviceVector<NumType>*>(&v_in);    
    DeviceVector<NumType>* w_out_d = dynamic_cast<DeviceVector<NumType>*>(w_out);
    assert(v_in_d != nullptr);
    assert(w_out_d != nullptr);

    const NumType zero = static_cast<NumType>(0);
    const NumType one = static_cast<NumType>(1);

    CUSPARSE_CALL( cusparseTcsrmv(manager::get_backend_struct().cusparseHandle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  this->m_,
                                  this->n_,
                                  this->nnz_,
                                  &one,
                                  this->cusparseMatDescr_,
                                  this->val_,
                                  this->row_ptr_,
                                  this->col_idx_,
                                  v_in_d->vec_,
                                  &zero,
                                  w_out_d->vec_));
}

template <typename NumType>
void DeviceMatrix<NumType>::compute_inverse_diagonals(BaseVector<NumType>* inv_diag) const
{
    assert(this->n_ > 0);
    assert(this->m_ > 0);
    assert(inv_diag != nullptr);

    DeviceVector<NumType>* inv_diag_d = dynamic_cast<DeviceVector<NumType>*>(inv_diag);
    assert(inv_diag_d != nullptr);

    if (inv_diag_d->n() != this->m_) {inv_diag_d->allocate(this->m_);
    }

    const int block = manager::get_backend_struct().dim_block_1d;
    const int grid = (this->m_ + block - 1)/block;
    
    compute_inverse_diag_kernel<<<grid, block>>>(this->m_,
                                                 this->row_ptr_,
                                                 this->col_idx_,
                                                 this->val_,
                                                 inv_diag_d->vec_);
}


template <typename NumType>
bool DeviceMatrix<NumType>::read_matrix_market(const std::string ) {
    Error("Method has not yet been implemented.");
}

// instantiate template classes
template class DeviceMatrix<double>;
template class DeviceMatrix<float>;
