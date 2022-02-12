#include "cublas_ctx.cuh"

void initialize_cublas(cublasHandle_t* handle) {
    
    cublasHandle_t handle_local[4];
    for(unsigned int i=0;i<4;i++) {
        cublasCreate_v2(&handle_local[i]);
        handle[i] = handle_local[i];
    }
    return;
}

void finalize_cublas(cublasHandle_t* handle) {
    for(unsigned int i=0;i<4;i++) {
        cublasDestroy_v2(handle[i]);
    }
    return;
}
