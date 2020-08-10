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
    CUBLAS_CALL( cublasCreate(&cublasHandle_) );
}

template <typename NumType>
DeviceVector<NumType>::~DeviceVector()
{
    CUBLAS_CALL( cublasDestroy(cublasHandle_) );
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
void DeviceVector<NumType>::copy_from(const NumType* w)
{
    assert(this->size_ > 0);

    CUDA_CALL(cudaMemcpy(this->vec_, w, this->size_ * sizeof(NumType), cudaMemcpyDeviceToDevice));
}

template <typename NumType>
void DeviceVector<NumType>::copy_from(const BaseVector<NumType>& w)
{
    assert(this != &w);
    
    const DeviceVector<NumType>* w_device = dynamic_cast<const DeviceVector<NumType>*>(&w);
    const HostVector<NumType>* w_host = dynamic_cast<const HostVector<NumType>*>(&w);
    
    // inputted vector is on device
    if (w_device != nullptr) {
        if (w_device->size_ != this->size_) {
            this->allocate(w_device->size_);
        }
        CUDA_CALL(cudaMemcpy(this->vec_, w_device->vec_, this->size_ * sizeof(NumType), cudaMemcpyDeviceToDevice));
    }
    // inputted vector is on host
    else if (w_host != nullptr) {
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
void DeviceVector<NumType>::copy_from_host(const NumType* w)
{
    assert(this->size_ > 0);

    CUDA_CALL(cudaMemcpy(this->vec_, w, this->size_ * sizeof(NumType), cudaMemcpyHostToDevice));
}

template <typename NumType>
void DeviceVector<NumType>::copy_to(NumType* w) const
{
    assert(this->size_ > 0);

    CUDA_CALL(cudaMemcpy(w, this->vec_, this->size_ * sizeof(NumType), cudaMemcpyDeviceToDevice));
}

template <typename NumType>
void DeviceVector<NumType>::copy_to(BaseVector<NumType>& w) const
{
    assert(this != &w);
    
    DeviceVector<NumType>* w_device = dynamic_cast<DeviceVector<NumType>*>(&w);
    HostVector<NumType>* w_host = dynamic_cast<HostVector<NumType>*>(&w);
    
    // inputted vector is on device
    if (w_device != nullptr) {
        if (w_device->size_ != this->size_) {
           w_device->allocate(this->size_);
        }
        CUDA_CALL(cudaMemcpy(w_device->vec_, this->vec_, this->size_ * sizeof(NumType), cudaMemcpyDeviceToDevice));
    }
    // inputted vector is on host
    else if (w_host != nullptr) {
        if (w_host->size_ != this->size_) {
            w_host->allocate(this->size_);
        }
        CUDA_CALL(cudaMemcpy(w_host->vec_, this->vec_, this->size_ * sizeof(NumType), cudaMemcpyDeviceToHost));
    }
    else {
        Error("Cannot cast vector as DeviceVector or HostVector.");
    }
}

template <typename NumType>
void DeviceVector<NumType>::copy_to_host(NumType* w) const
{
    assert(this->size_ > 0);

    CUDA_CALL(cudaMemcpy(w, this->vec_, this->size_ * sizeof(NumType), cudaMemcpyDeviceToHost));
}

template <typename NumType>
NumType& DeviceVector<NumType>::operator[](int i)
{
    Error("Method has not yet been implemented.");
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
    thrust::fill(thrust_dev_ptr, thrust_dev_ptr + this->size_, zero);
}

template <typename NumType>
void DeviceVector<NumType>::ones() 
{
    const NumType one = static_cast<NumType>(1);
    thrust::device_ptr<NumType> thrust_dev_ptr(this->vec_);
    thrust::fill(thrust_dev_ptr, thrust_dev_ptr + this->size_, one);
}

template <typename NumType>
void DeviceVector<NumType>::scale(NumType alpha)
{
    assert(this->size_ > 0);
    CUBLAS_CALL( cublasTscal(cublasHandle_,
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

    add_kernel<<<grid, block>>>(this->size_,
                               alpha,
                               beta,
                               w_device->vec_,
                               this->vec_);
    CUDA_CALL( cudaDeviceSynchronize());
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
    
    scale_add_kernel<<<grid, block>>>(this->size_,
                               alpha,
                               w_device->vec_,
                               this->vec_);
    CUDA_CALL( cudaDeviceSynchronize());
}

// instantiate template classes
template class DeviceVector<double>;
template class DeviceVector<float>;
