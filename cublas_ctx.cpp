#include "cublas_ctx.cuh"

void initialize_cublas(cublasHandle_t* handle) {
    
    for(unsigned int i=0;i<4;i++) {
        cublasCreate_v2(&handle[i]);
    }
    return;
}

void finalize_cublas(cublasHandle_t* handle) {
    for(unsigned int i=0;i<4;i++) {
        cublasDestroy_v2(handle[i]);
    }
    return;
}
