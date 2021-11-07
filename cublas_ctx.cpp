#include "cublas_ctx.cuh"

void initialize_cublas(cublasHandle_t* handle) {
    cublasCreate_v2(handle);
    return;
}

void finalize_cublas(cublasHandle_t* handle) {
    cublasDestroy_v2(*handle);
    return;
}
