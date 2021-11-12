#include "cuda_runtime.h"
#include <cmath>
#include "gmres_ctx.cuh"
/** an example to do matrix-vector product
 *  3 2
 *  1 3 2
 *    1 3 2
 *      1 3 2
 *        ...
 *        1 3
 */


template<typename T>
__global__
void MatDotVec(T* c, T* b, unsigned int n) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(idx >= n) return;
    if(idx == 0) c[idx] = 3.0*b[idx] + 1.2*b[idx+1];
    else
    if(idx == n-1 ) c[idx] = 1.1*b[idx-1]+3.0*b[idx];
    else c[idx] = 1.1*b[idx-1]+3.0*b[idx]+1.2*b[idx+1];
}

template<typename T>
void MatDotVec_wrapper(T* c, T* b, unsigned int xdim) {
    
    unsigned int threads_per_block = 256;
    unsigned int blocks = ceil(float(xdim)/256);

    MatDotVec<T><<<blocks, threads_per_block >>>(c, b, xdim);
}


__global__
void jacobiPreconditioning(double* X, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>= xdim) return;
    X[idx] = 3* X[idx];
}

template<typename T> 
void MFPreconditioner(void *mfprecon, T* X) {
    struct precon_app_ctx* pctx = (struct precon_app_ctx*) mfprecon;
    unsigned int xdim = pctx->xdim;
    unsigned int threads_per_block = 256;
    unsigned int blocks = ceil(float(xdim)/256);

    jacobiPreconditioning<<<blocks,threads_per_block>>>(X, xdim);
    return;
};


