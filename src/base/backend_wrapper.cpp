
#include "base/backend_wrapper.h"

BackendInfoStruct Backend {
    0,
    0,
    256
};
namespace manager {
    void start_backend() {    
        CUBLAS_CALL( cublasCreate(&(Backend.cublasHandle) ));
        CUSPARSE_CALL( cusparseCreate(&(Backend.cusparseHandle) ));
    }
    
    void stop_backend() {
        CUBLAS_CALL( cublasDestroy(Backend.cublasHandle) );
        CUSPARSE_CALL( cusparseDestroy(Backend.cusparseHandle) );
    }

    BackendInfoStruct get_backend_struct() {
        return Backend;
    }
}