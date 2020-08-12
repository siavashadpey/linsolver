
#include "base/backend_wrapper.h"

namespace manager {
    void start_backend() {
        Backend.dim_block_1d = 256;
    
        CUBLAS_CALL( cublasCreate(&(Backend.cublasHandle) ));
        CUSPARSE_CALL( cusparseCreate(&(Backend.cusparseHandle) ));
    }
    
    void stop_backend() {
        CUBLAS_CALL( cublasDestroy(Backend.cublasHandle) );
        CUSPARSE_CALL( cusparseDestroy(Backend.cusparseHandle) );
    }
}