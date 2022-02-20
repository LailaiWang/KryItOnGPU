#ifndef __CUBLAS__CTX__
#define __CUBLAS__CTX__

#include "cuda_runtime.h"
#include "cublas_v2.h"

void initialize_cublas(cublasHandle_t* handle);
void finalize_cublas(cublasHandle_t* handle);

struct cublas_app_ctx {
    
    /*create four handles*/
    cublasHandle_t handle;
    
    /*streams for first queue in PyFR*/
    cudaStream_t stream;
    
    cublas_app_ctx(
            cudaStream_t stream_comp_0,
            void (*creat) (cublasHandle_t*),
            void (*clean) (cublasHandle_t*)
    ) {
        
        stream = stream_comp_0;

        create_cublas = creat;
        clean_cublas  = clean;

    }

    void (*create_cublas) (cublasHandle_t* );
    void (*clean_cublas)  (cublasHandle_t* );
};


#endif
