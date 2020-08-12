#ifndef BACKEND_WRAPPER_H
#define BACKEND_WRAPPER_H

#include "base/cuda_header.cuh"

struct BackendInfoStruct {
    cublasHandle_t   cublasHandle;
    cusparseHandle_t cusparseHandle;
    int              dim_block_1d;
};

namespace manager {
    void start_backend();
    void stop_backend();
}

extern struct BackendInfoStruct Backend;

#endif