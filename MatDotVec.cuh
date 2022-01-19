#include "cuda_runtime.h"
#include <cmath>
//#include "gmres_ctx.cuh"

template<typename T>
__global__
void MatDotVec(T* c, T* b, unsigned int n) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(idx >= n) return;
    if(idx == 0) c[idx] = 2.0*b[idx] + 1.0*b[idx+1];
    else
    if(idx == n-1 ) c[idx] = 1.0*b[idx-1]+2.0*b[idx];
    else c[idx] = 1.0*b[idx-1]+2.0*b[idx]+1.0*b[idx+1];

    printf("c[%d] %lf b[%d] %lf\n", idx, c[idx], idx, b[idx]);
}

template<typename T>
void MatDotVec_wrapper(void* c, void* b, unsigned int xdim) {
    
    unsigned int threads_per_block = 256;
    unsigned int blocks = ceil(float(xdim)/256);
    
    T* cptr = (T*) c;
    T* bptr = (T*) b;

    MatDotVec<T><<<blocks, threads_per_block >>>(cptr, bptr, xdim);
}


__global__
void jacobiPreconditioning(double* X, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>= xdim) return;
    X[idx] = 3* X[idx];
}

template<typename T> 
void MFPreconditioner(void *mfprecon, T* X) {
    struct precon_app_ctx<T>* pctx = (struct precon_app_ctx<T>*) mfprecon;
    unsigned int xdim = pctx->xdim;
    unsigned int threads_per_block = 256;
    unsigned int blocks = ceil(float(xdim)/256);

    jacobiPreconditioning<<<blocks,threads_per_block>>>(X, xdim);
    return;
};


