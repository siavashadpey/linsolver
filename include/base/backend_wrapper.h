#ifndef BACKEND_WRAPPER_H
#define BACKEND_WRAPPER_H

#include "base/cuda_header.cuh"

struct BackendInfoStruct {
    cublasHandle_t   cublasHandle;
    cusparseHandle_t cusparseHandle;
    int              dim_block_1d;

    extern struct BackendInfoStruct Backend;

    void start_backend() {
        BackendInfo.dim_block_1d = 256;

        CUBLAS_CALL( cublasCreate(&(Backend.cublasHandle) ));
        CUSPARSE_CALL( cusparseCreate(&(Backend.cusparseHandle) ));
    }

    void stop_backend() {
        CUBLAS_CALL( cublasDestroy(Backend.cublasHandle) );
        CUSPARSE_CALL( cusparseDestroy(Backend.cusparseHandle) );
    }

}

#endif