#ifndef __CUBLAS__CTX__
#define __CUBLAS__CTX__

#include "cuda_runtime.h"
#include "cublas_v2.h"

void initialize_cublas(cublasHandle_t* handle);
void finalize_cublas(cublasHandle_t* handle);

struct cublas_app_ctx {
    cublasHandle_t handle;
    
    /* constructor using standalone functions to initialize function 
     * pointers
     */
    cublas_app_ctx(void (*creat) (cublasHandle_t*), void (*clean) (cublasHandle_t*)) {
        create_cublas = creat;
        clean_cublas  = clean;
    }

    void (*create_cublas) (cublasHandle_t* );
    void (*clean_cublas)  (cublasHandle_t* );
};


#endif
