#include <stdio.h>
#include <cmath>
#include <cassert>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "base/cuda_header.cuh"
#include "base/cublas_wrapper.cuh"
#include "backends/device/device_vector_helper.cuh"
#include "base/error.h"

#include "backends/device/device_vector.h"
#include "backends/host/host_vector.h"

template <typename NumType>
DeviceVector<NumType>::DeviceVector()
    :   vec_(nullptr)
{
    CUBLAS_CALL( cublasCreate(&cublasHandle) );
}

template <typename NumType>
DeviceVector<NumType>::~DeviceVector()
{
    CUBLAS_CALL( cublasDestroy(&cublasHandle) );
    this->clear();
}

template <typename NumType>
void DeviceVector<NumType>::allocate(int n)
{
    assert(n > 0);
    this->clear();

    this->size_ = n;
    CUDA_CALL(cudaMalloc((void**) &(this->vec_), n * sizeof(NumType)));
}

template <typename NumType>
void DeviceVector<NumType>::clear()
{
    if (this->size_ > 0) {
        CUDA_CALL(cudaFree(this->vec_));
        this->size_ = 0;
    }
}

template <typename NumType>
void DeviceVector<NumType>::copy(const NumType* w)
{
    assert(this->size_ > 0);

    CUDA_CALL(cudaMemcpy(this->vec_, w, this->size_ * sizeof(NumType), cudaMemcpyDeviceToDevice));
}

template <typename NumType>
void DeviceVector<NumType>::copy(const BaseVector<NumType>& w)
{
    assert(this != &w);
    
    const DeviceVector<NumType>* w_device = dynamic_cast<const DeviceVector<NumType>*>(&w);
    const HostVector<NumType>* w_host = dynamic_cast<const HostVector<NumType>*>(&w);
    
    // inputted vector is on device
    if (w_device != nullptr) {
        printf("should not be here\n");
        if (w_device->size_ != this->size_) {
            this->allocate(w_device->size_);
        }
        CUDA_CALL(cudaMemcpy(this->vec_, w_device->vec_, this->size_ * sizeof(NumType), cudaMemcpyDeviceToDevice));
    }
    // inputted vector is on host
    else if (w_host != nullptr) {
        printf("should be here\n");
        if (w_host->size_ != this->size_) {
            this->allocate(w_host->size_);
        }
        CUDA_CALL(cudaMemcpy(this->vec_, w_host->vec_, this->size_ * sizeof(NumType), cudaMemcpyHostToDevice));
    }
    else {
        Error("Cannot cast vector as DeviceVector or HostVector.");
    }
}

template <typename NumType>
NumType& DeviceVector<NumType>::operator[](int i)
{
    Error("Method is not currently supported");
    return static_cast<NumType>(1);
}

template <typename NumType>
NumType DeviceVector<NumType>::norm() const
{
    NumType val = static_cast<NumType>(0);
    assert(this->size_ > 0);

    CUBLAS_CALL( cublasTnrm2(cublasHandle_, 
                             this->size_, 
                             this->vec_, 
                             1,             // increment of 1
                             &val));
    return val;
}

template <typename NumType>
NumType DeviceVector<NumType>::dot(const BaseVector<NumType>& w) const
{
    const DeviceVector<NumType>* w_device = dynamic_cast<const DeviceVector<NumType>*>(&w);
    assert(w_device != nullptr);
    assert(this->size_ == w_device->size_);

    NumType val = static_cast<NumType>(0);

    CUBLAS_CALL( cublasTdot(cublasHandle_,
                            this->size_, 
                            this->vec_, 1, // increment of 1
                            w_device->vec_, 1, // increment of 1 
                            &val));
    return val;
}

template <typename NumType>
void DeviceVector<NumType>::zeros() 
{
    const NumType zero = static_cast<NumType>(0);
    thrust::device_ptr<NumType> thrust_dev_ptr(this->vec_);
    thrust::fill(this->vec_, this->vec_ + this->size_, zero);
}

template <typename NumType>
void DeviceVector<NumType>::ones() 
{
    const NumType one = static_cast<NumType>(1);
    thrust::device_ptr<NumType> thrust_dev_ptr(this->vec_);
    thrust::fill(this->vec_, this->vec_ + this->size_, one);
}

template <typename NumType>
void DeviceVector<NumType>::scale(NumType alpha)
{
    assert(this->size_ > 0);
    CUBLASS_CALL( cublasTscal(cublasHandle_,
                              this->size_, 
                              &alpha,
                              this->vec_, 1 // increment of 1
                              ));
}

template <typename NumType>
void DeviceVector<NumType>::add(NumType alpha, const BaseVector<NumType>& w, NumType beta)
{
    const NumType one = static_cast<NumType>(1);

    if (alpha == one)
    {
        this->add_scale(beta, w);
        return;
    }
    else if (beta == one)
    {
        this->scale_add(alpha, w);
        return;
    }

    const DeviceVector<NumType>* w_device = dynamic_cast<const DeviceVector<NumType>*>(&w);
    assert(w_device != nullptr);
    assert(this->size_ == w_device->size_);


    const int block = 256;
    const int grid = (this->size_ + block - 1)/block;

    add<<<grid, block>>>(this->size_,
                               &alpha,
                               &beta,
                               w_device->vec_,
                               this->vec_);
}

template <typename NumType>
void DeviceVector<NumType>::add_scale(NumType alpha, const BaseVector<NumType>& w)
{
    const NumType zero = static_cast<NumType>(0);
    if (alpha == zero) {
        return;
    }
    
    const DeviceVector<NumType>* w_device = dynamic_cast<const DeviceVector<NumType>*>(&w);
    assert(w_device != nullptr);
    assert(this->size_ == w_device->size_);

    CUBLAS_CALL( cublasTaxpy(cublasHandle_,
                             this->size_,
                             &alpha,
                             w_device->vec_, 1, // increment of 1
                             this->vec_, 1 // increment of 1
                             ));
}

template <typename NumType>
void DeviceVector<NumType>::scale_add(NumType alpha, const BaseVector<NumType>& w)
{
    const DeviceVector<NumType>* w_device = dynamic_cast<const DeviceVector<NumType>*>(&w);
    assert(w_device != nullptr);
    assert(this->size_ == w_device->size_);

    const int block = 256;
    const int grid = (this->size_ + block - 1)/block;
    
    scale_add<<<grid, block>>>(this->size_,
                               &alpha,
                               w_device->vec_,
                               this->vec_);
    // TODO: check error?
}

// instantiate template classes
template class DeviceVector<double>;
template class DeviceVector<float>;
