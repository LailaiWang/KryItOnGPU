#ifndef __CUBLAS__CTX__
#define __CUBLAS__CTX__

#include "cuda_runtime.h"
#include "cublas_v2.h"

void initialize_cublas(cublasHandle_t* handle);
void finalize_cublas(cublasHandle_t* handle);

struct cublas_app_ctx {
    
    /*create four handles*/
    cublasHandle_t handle[4];
    
    /*streams for first queue in PyFR*/
    cudaStream_t comp_stream_0;
    cudaStream_t copy_stream_0;

    /*streams for second queue in PyFR*/
    cudaStream_t comp_stream_1;
    cudaStream_t copy_stream_1;
    
    cublas_app_ctx(
            cudaStream_t stream_comp_0,
            cudaStream_t stream_copy_0,
            cudaStream_t stream_comp_1,
            cudaStream_t stream_copy_1,
            void (*creat) (cublasHandle_t*),
            void (*clean) (cublasHandle_t*)
    ) {
        
        comp_stream_0 = stream_comp_0;
        copy_stream_0 = stream_copy_0;

        comp_stream_1 = stream_comp_1;
        copy_stream_1 = stream_copy_1;

        create_cublas = creat;
        clean_cublas  = clean;

    }

    void (*create_cublas) (cublasHandle_t* );
    void (*clean_cublas)  (cublasHandle_t* );
};


#endif
