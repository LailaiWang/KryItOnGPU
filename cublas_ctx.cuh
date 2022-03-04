#ifndef __CUBLAS__CTX__
#define __CUBLAS__CTX__

#include "cuda_runtime.h"
#include "cublas_v2.h"

struct cublas_app_ctx {
    /*create four handles*/
    cublasHandle_t handle[2];
    /*streams for first queue in PyFR*/
    cudaStream_t stream;
    /* small constructor here*/
    cublas_app_ctx( cudaStream_t stream_comp_0) {
        stream = stream_comp_0;
        /*create two handles*/
        cublasCreate_v2(&handle[0]);
        cublasCreate_v2(&handle[1]);
        /*assign stream to handle*/
        cublasSetStream(handle[0], stream);
        cublasSetStream(handle[1], 0);
        /*set pointer mode*/
        cublasSetPointerMode(handle[0], CUBLAS_POINTER_MODE_DEVICE);
        cublasSetPointerMode(handle[1], CUBLAS_POINTER_MODE_DEVICE);
    }
    /* small destructor here*/
    ~cublas_app_ctx() {
        cublasDestroy(handle[0]);
        cublasDestroy(handle[1]);
    }
};


#endif
