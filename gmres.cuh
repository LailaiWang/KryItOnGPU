/* GMRES solver on GPU with matrix-free implementation 
 * Matrix-free matrix-vector product
 * Matrix-free preconditioner
 */ 

#ifndef GMRES_CUH
#define GMRES_CUH

#include "arnoldi.cuh"
#include "gmres_ctx.cuh"
#include "cublas_ctx.cuh"
#include "cublas_v2.h"
#include "math.h"

__global__
void set_zero_double(double* x, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < xdim) x[idx] = 0.0;
}

__global__
void set_zero_float(float* x, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < xdim) x[idx] = 0.0f;
}

/** 
 * \fn a function to set array to zero on the device
 *
 */
template<typename T>
void set_zero_wrapper(T* x, unsigned int xdim) {
    unsigned int threads_per_block = 256;
    unsigned int blocks = ceil(float(xdim)/threads_per_block);
    if constexpr (std::is_same<float, T>::value) {
        set_zero_float<<<blocks, threads_per_block>>>(x, xdim);
    } else
    if constexpr (std::is_same<double, T>::value){
        set_zero_double<<<blocks, threads_per_block>>>(x, xdim);
    }
}

/** 
 * \fn gmres function
 *
 */
template<typename T>
void MFgmres(void (*MatDotVec) (T*, T*, unsigned int), // matrix-vector product
           void (*preconditioner) (void*), // preconditioner
           unsigned long long int ptx,
           unsigned long long int ctx,
           unsigned long long int btx
          ) {

    // cast the void pointer into a application context pointer
    cublasStatus_t err;
    
    // grab different application context
    struct precon_app_ctx* pcon_ctx =
        reinterpret_cast<struct precon_app_ctx*> (ptx);
    struct gmres_app_ctx* gmres_ctx = 
        reinterpret_cast<struct gmres_app_ctx*> (ctx);
    struct cublas_app_ctx* blas_ctx = 
        reinterpret_cast<struct cublas_app_ctx*> (btx);
    
    // dimension of the problem and krylov subspace
    unsigned int xdim = gmres_ctx->xdim;
    unsigned int kspace = gmres_ctx->kspace;

    T* sn   = gmres_ctx->sn;    // dimension kspace
    T* cs   = gmres_ctx->cs;    // dimension kspace
    T* e1   = gmres_ctx->e1;    // dimension kspace+1
    T* beta = gmres_ctx->beta;  // dimension kspace+1
    
    T* b = gmres_ctx->b; // dimension xdim Ax = b 
    T* Q = gmres_ctx->Q; // dimension xdim*(kspace+1)
    T* h = gmres_ctx->h; // dimension (kspace+1)*kspace
    T* v = gmres_ctx->v; // dimension  xdim

    
    T* x   = pcon_ctx->x;
    T* res = pcon_ctx->res;

    T atol = gmres_ctx->atol;
    T rtol = gmres_ctx->rtol;
    
    set_zero_wrapper(sn,   kspace+1);
    set_zero_wrapper(cs,   kspace+1);
    set_zero_wrapper(e1,   kspace+1);
    set_zero_wrapper(beta, kspace+11);
    
    /*initialize guess all zeros*/
    set_zero_wrapper(x, xdim);
    
    /*first arg: output second: input*/
    MatDotVec(res, x, xdim);
    /*
    cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
                           const float *alpha,
                           const float *x, int incx,
                           float       *y, int incy)
    cublasStatus_t  cublasDscal(cublasHandle_t handle, int n,
                            const double *alpha,
                            double       *x, int incx)
    */
    T one = 1.0;
    cudaMemcpy(e1, &one, sizeof(T), cudaMemcpyHostToDevice);

    if constexpr (std::is_same<float, T>::value) {
        float alpha = -1.0f;
        err = cublasSscal(blas_ctx->handle, xdim, &alpha, res, 1);
        alpha = 1.0f;
        err = cublasSaxpy(blas_ctx->handle, xdim, &alpha, b, 1, res, 1);
    } else 
    if constexpr (std::is_same<double, T>::value) {
        double alpha = -1.0;
        err = cublasDscal(blas_ctx->handle, xdim, &alpha, res, 1);
        alpha = 1.0;
        err = cublasDaxpy(blas_ctx->handle, xdim, &alpha, b, 1, res, 1);
    }
    /*
    cublasStatus_t  cublasSnrm2(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result)
    cublasStatus_t  cublasDnrm2(cublasHandle_t handle, int n,
                            const double          *x, int incx, double *result)
    */
    T bnorm;
    T rnorm;

    if constexpr (std::is_same<float, T>::value) {
        err = cublasSnrm2(blas_ctx->handle, xdim, b,   1, &bnorm);
        err = cublasSnrm2(blas_ctx->handle, xdim, res, 1, &rnorm);
    } else
    if constexpr (std::is_same<double, T>::value){
        err = cublasDnrm2(blas_ctx->handle, xdim, b,   1, &bnorm);
        err = cublasDnrm2(blas_ctx->handle, xdim, res, 1, &rnorm);
    }
    
    T error = bnorm/rnorm;
    T error0 = error;

    if(error0 < atol) return; /*already converged, directly return*/

    if(std::isnan(error)) {
        printf("GMRES norm inf\n");
        exit(0);
    }

    /*
    cublasStatus_t cublasScopy(cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           float                 *y, int incy)
    */

    if constexpr (std::is_same<float, T>::value) {
        float rnormi = 1.0f/rnorm;
        err = cublasScopy(blas_ctx->handle, xdim, res, 1, Q, 1);
        err = cublasSscal(blas_ctx->handle, xdim, &rnormi, Q, 1);
        // beta vector 
        err = cublasScopy(blas_ctx->handle, xdim, e1, 1, beta, 1);
        err = cublasSscal(blas_ctx->handle, xdim, &rnorm, beta, 1);
    } else
    if constexpr (std::is_same<double, T>::value){
        double rnormi = 1.0/rnorm;
        err = cublasDcopy(blas_ctx->handle, xdim, res,1, Q, 1);
        err = cublasDscal(blas_ctx->handle, xdim, &rnormi, Q, 1);
        // beta vector
        err = cublasDcopy(blas_ctx->handle, xdim, e1,1, beta, 1);
        err = cublasDscal(blas_ctx->handle, xdim, &rnorm, beta, 1);
    }
    
    /*some integer representation of the address*/
    unsigned long long int Q0 = reinterpret_cast<unsigned long long int> (Q);
    unsigned long long int h0 = reinterpret_cast<unsigned long long int> (h);
    unsigned long long int c0 = reinterpret_cast<unsigned long long int> (cs);
    unsigned long long int s0 = reinterpret_case<unsigned long long int> (sn);

    for(unsigned int k=0;k<kspace;k++) {
        
        MatDotVec(v, reinterpret_cast<T*>(Q0+sizeof(T)*xdim*k), xdim); 
        for(unsigned int j=0;j<k;j++) {
            // since cols of the matrix are stored continuously
            // the offset of hjk is k*(kspace+1) + j (h is a (kspace+1)*kspace matrix)
            T* hjk = reinterpret_cast<T*> (h0+sizeof(T)*(j+k*(kspace+1)));
            // the offset of Qj is  since Qj is a xdim*kspace matrix
            T* Qj  = reinterpret_cast<T*> (Q0+sizeof(T)*(xdim*j));
            /*
            cublasStatus_t cublasSdot (cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float                 *result)
            cublasStatus_t cublasDdot (cublasHandle_t handle, int n,
                           const double          *x, int incx,
                           const double          *y, int incy,
                           double                *result)
            */

            if constexpr (std::is_same<float, T>::value) {
                err = cublasSdot(blas_ctx->handle, h, 1, v, 1, hjk);
                err = cublasSaxpy(blas_ctx->handle, xdim, hjk, 1, Q, 1);
            } else 
            if constexpr (std::is_same<double,T>::value) {
                // hjk = dot(v, Q)
                err = cublasSdot(blas_ctx->handle, v, 1, Q, 1, hjk);
                // update v = v-hjk*Qj
                err = cublasSaxpy(blas_ctx->handle, xdim, hjk, 1, Q, 1);
            }

        }
        // update h(k+1,k)
        
        T* hkp1k = reinterpret_cast<T*> (h0+sizeof(T)*((k)+(k-1)*(kspace+1)));

        T vnorm;

        if constexpr (std::is_same<float, T>::value) {
            err = cublasSnrm2(blas_ctx->handle, xdim, v,   1, &vnorm);
        } else 
        if constexpr (std::is_same<double, T>::value) {
            err = cublasDnrm2(blas_ctx->handle, xdim, v,   1, &vnorm);
        }

        // copy the value to vnorm
        cudaMemcpy(hkp1k, &vnorm, sizeof(T), cudaMemcpyHostToDevice);
        // normalize v
        if constexpr (std::is_same<float, T>::value) {
            float vnormi = 1.0f/vnorm;
            err = cublasDscal(blas_ctx->handle, xdim, &vnormi, v, 1);
        } else 
        if constexpr (std::is_same<double, T>::value) {
            double vnormi = 1.0/vnorm;
            err = cublasDscal(blas_ctx->handle, xdim, &vnormi, v, 1);
        }


        /*
        cublasStatus_t cublasSrotg(cublasHandle_t handle,
                           float  *a, float *b,
                           float  *c, float *s)
        cublasStatus_t cublasDrotg(cublasHandle_t handle,
                           double *a, double *b,
                           double *c, double *s)
        */
        if constexpr (std::is_same<float, T>::value) {
            err = cublasSrotg(
                blas_ctx->handle,
                h,
                h,
                cs,
                sn
            );
        } else 
        if constexpr (std::is_same<float, T>::value) {
            err = cublasDrotg(
                blas_ctx->handle,
                h,
                h,
                cs,
                sn
            );
        }
        
        // beta (k+1) = -sn(k)*beta(k)
        // beta (k  ) =  cs(k)*beta(k)
        // error       = abs(beta(k + 1)) / b_norm;
        error = 0.0;
        
        if(error < atol || error/error0 < rtol) {
            break;
        }
    }

    // calculate the result
    //y = H(1:k, 1:k) \ beta(1:k);
    //x = x + Q(:, 1:k) * y;

    if constexpr (std::is_same<float, T>::value) {
        
    } else 
    if constexpr (std::is_same<double, T>::value) {


    }
}

#endif
