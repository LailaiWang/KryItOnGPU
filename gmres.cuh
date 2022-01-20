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
#include "util.cuh"
#include "cuda_constant.cuh"
#include "Python.h"

__global__
void givens_rot_d(double* h1, double* h2, double* c, double* s) {
    double tmp = sqrt((*h1)*(*h1)+(*h2)*(*h2));
    *c = *h1/tmp;
    *s = *h2/tmp;
    *h1 = (*c)*(*h1)+(*s)*(*h2);
    *h2 = 0.0f;
    return;
}

__global__
void apply_rot_d(double* h, double* cs, double* sn, unsigned int k) {
    double temp = 0.0f;
    for(unsigned int i=0;i<k;i++) {
        temp = cs[i]*h[i]+sn[i]*h[i+1];
        h[i+1] = -sn[i]*h[i]+cs[i]*h[i+1];
        h[i  ] = temp;
    }
    return;
}

__global__
void apply_rot_beta_d(double* beta, double* cs, double* sn, unsigned int k) {
    double temp = 0.0f;
    for(unsigned int i=0;i<k;i++) {
        temp = cs[i]*beta[i]+sn[i]*beta[i+1];
        beta[i+1] = -sn[i]*beta[i]+cs[i]*beta[i+1];
        beta[i  ] = temp;
    }
    return;
}

__global__
void apply_rot_beta_f(float* beta, float* cs, float* sn, unsigned int k) {
    float temp = 0.0f;
    for(unsigned int i=0;i<k;i++) {
        temp = cs[i]*beta[i]+sn[i]*beta[i+1];
        beta[i+1] = -sn[i]*beta[i]+cs[i]*beta[i+1];
        beta[i  ] = temp;
    }
    return;
}

__global__
void givens_rot_f(float* h1, float* h2, float* c, float* s) {
    float tmp = sqrt((*h1)*(*h1)+(*h2)*(*h2));
    *c = *h1/tmp;
    *s = *h2/tmp;
    *h1 = (*c)*(*h1)+(*s)*(*h2);
    *h2 = 0.0f;
    return;
}

__global__
void apply_rot_f(float* h, float* cs, float* sn, unsigned int k) {
    float temp = 0.0f;
    for(unsigned int i=0;i<k;i++) {
        temp = cs[i]*h[i]+sn[i]*h[i+1];
        h[i+1] = -sn[i]*h[i]+cs[i]*h[i+1];
        h[i  ] = temp;
    }
    return;
}


__global__
void dot_one_d(double* a, double* b, double* c, double scal) {
    *c = (*a)*(*b)*scal;
    return;
}

__global__
void dot_one_f(float* a, float* b, float* c, float scal) {
    *c = (*a)*(*b)*scal;
    return;
}


template<typename T>
auto get_addr(unsigned long long int start, unsigned int offset) {

    return reinterpret_cast<T*> (start+sizeof(T)*offset);
}


__global__
void get_soln_d(double* wts, double* x, double* q, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= xdim) return;
    x[idx] += (*wts)*q[idx];
    return;
}

__global__
void get_soln_f(float* wts, float* x, float* q, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= xdim) return;
    x[idx] += (*wts)*q[idx];
    return;
}

/** 
 * \fn a function to set array to zero on the device
 *
 */
template<typename T>
void set_zero_wrapper(T* x, unsigned int xdim) {
    if constexpr (std::is_same<float, T>::value) {
        set_zeros_float(x, xdim);
    } else
    if constexpr (std::is_same<double, T>::value){
        set_zeros_double(x, xdim);
    }
}

/*norm for multiple gpus*/
template<typename T> 
void mGPU_norm2_wrapper(cublasHandle_t handle, unsigned int xdim, T* vec, T* vnorm) {
    cublasStatus_t err;
    T localnorm;
    if constexpr (std::is_same<float,T>::value) {
        err = cublasSdot(handle, xdim,vec, 1, vec, 1, &localnorm);
    } else 
    if constexpr (std::is_same<double, T>:: value) {
        err = cublasDdot(handle, xdim,vec, 1, vec, 1, &localnorm);
    }
#ifdef _MPI
    MPI_datatype dtype = MPI_DATATYPE_NULL;

    if constexpr (std::is_same<float, T>::value) {
        dtype = MPI_FLOAT;
    } else 
    if constexpr (std::is_same<double, T>::value){
        dtype = MPI_DOUBLE;
    } 
    // sum value into bnorm[1]
    MPI_Allreduce(vnorm, &localnorm, 1, dtype, MPI_SUM, MPI_COMM_WORLD);
    *vnorm = std::sqrt(*vnorm);
#else
    *vnorm = std::sqrt(localnorm);
#endif
    return;
}

/*dot product for multiple gpus*/
template<typename T>
void mGPU_dot_wrapper(cublasHandle_t handle, unsigned int xdim, T* vec, T* vnorm) {
    cublasStatus_t err;
    T localnorm;
    if constexpr (std::is_same<float,T>::value) {
        err = cublasSdot(handle, xdim,vec, 1, vec, 1, &localnorm);
    } else 
    if constexpr (std::is_same<double, T>:: value) {
        err = cublasDdot(handle, xdim,vec, 1, vec, 1, &localnorm);
    }
#ifdef _MPI
    MPI_datatype dtype = MPI_DATATYPE_NULL;

    if constexpr (std::is_same<float, T>::value) {
        dtype = MPI_FLOAT;
    } else 
    if constexpr (std::is_same<double, T>::value){
        dtype = MPI_DOUBLE;
    } 
    // sum value into bnorm[1]
    MPI_Allreduce(vnorm, &localnorm, 1, dtype, MPI_SUM, MPI_COMM_WORLD);
#else
    *vnorm = localnorm;
#endif
    return;
}

/** 
 * \fn gmres function
 *  The bottomneck of gmres method is matrix-vector product
 *  We use a matrix free implementation for single GPU multiple GPU cases
 *  however, the least square problem in gmres is so tiny (especially when we have a small
 *  krylov subspace), such that
 *  cross GPU computation is not needed (would make things slower). Hence,
 *  We only use single GPU to calculate it (well each GPU can do the same 
 *  calculation simultaneously). With this being said,
 *  this only thing we need to worry about cross GPUs is norm2 calculation
 *  Other than that, everything can directly taking adcantage of local cublas 
 *  calculations.
 */
template<typename T>
void MFgmres(
      void (*MatDotVec) (void*, void*, void*, unsigned int), /* matrix-vector product func*/
      void (*preconditioner) (void*, void*, unsigned int), /* preconditioner func*/
      void* solctx, /*solver context defined as provided*/
      void* ptx,     
      void* gtx,
      void* btx
    ) {
    // grab different application context
    struct precon_app_ctx<T>* pcon_ctx = (struct precon_app_ctx<T>*) (ptx);
    struct gmres_app_ctx<T>* gmres_ctx = (struct gmres_app_ctx<T>*) (gtx);
    struct cublas_app_ctx* blas_ctx = (struct cublas_app_ctx*) (btx);

    // dimension of the problem and krylov subspace
    unsigned int xdim = gmres_ctx->xdim;
    unsigned int kspace = gmres_ctx->kspace;

    T* sn   = gmres_ctx->sn;    // dimension kspace+1
    T* cs   = gmres_ctx->cs;    // dimension kspace+1
    T* e1   = gmres_ctx->e1;    // dimension kspace+1
    T* beta = gmres_ctx->beta;  // dimension kspace+11
    
    T* b = gmres_ctx->b; // dimension xdim Ax = b 
    T* Q = gmres_ctx->Q; // dimension xdim*(kspace+1)
    T* h = gmres_ctx->h; // dimension (kspace+1)*kspace
    T* v = gmres_ctx->v; // dimension  xdim

    
    T* x   = pcon_ctx->x;
    T* res = pcon_ctx->res;

    T atol = gmres_ctx->atol;
    T rtol = gmres_ctx->rtol;
    
    set_zero_wrapper(sn,   kspace+1);   /*initialization to 0*/
    set_zero_wrapper(cs,   kspace+1);
    set_zero_wrapper(e1,   kspace+1);
    set_zero_wrapper(beta, kspace+11);
    
    set_zero_wrapper(x, xdim); /*initialize guess all zeros*/
    
    bool restart = false;  /*if it is restart or not*/

    unsigned int cnt = 0;  /*reset when restart, dimension of the least square problem*/

    gmres_ctx->conv_iters = 0;  /**/

    T error  = 0.0; /*residual*/
    T error0 = 0.0; 

    T bnorm;
    T rnorm;
    
    RESTART_ENTRY: {
        cnt = 0;  /*no. of iterations till convergence need to reset when restart*/
        /*perform matrix vector product here*/
        MatDotVec(solctx, (void*) res, (void*) x, xdim);   /*first arg: output second: input*/
    }

    /*r = b - Ax_0*/
    if constexpr (std::is_same<float, T>::value) {
        CUDA_CALL(cudaMemcpyFromSymbol(e1, P_ONE_F,sizeof(float), 0, cudaMemcpyDeviceToDevice);)
        cublasSscal(blas_ctx->handle, xdim, &N_1F, res, 1);
        cublasSaxpy(blas_ctx->handle, xdim, &P_1F, b, 1, res, 1);
    } else 
    if constexpr (std::is_same<double, T>::value) {
        CUDA_CALL(cudaMemcpyFromSymbol(e1, P_ONE_D,sizeof(double), 0, cudaMemcpyDeviceToDevice);)
        cublasDscal(blas_ctx->handle, xdim, &N_1D, res, 1);
        cublasDaxpy(blas_ctx->handle, xdim, &P_1D, b, 1, res, 1);
    }
    
    if constexpr (std::is_same<float, T>::value) {
        cublasSnrm2(blas_ctx->handle, xdim, b,   1, &bnorm);
        cublasSnrm2(blas_ctx->handle, xdim, res, 1, &rnorm);
    } else
    if constexpr (std::is_same<double, T>::value){
        cublasDnrm2(blas_ctx->handle, xdim, b,   1, &bnorm);
        cublasDnrm2(blas_ctx->handle, xdim, res, 1, &rnorm);
    }
    
    error = rnorm/bnorm;
    if (restart == false ) error0 = error; /*only update error0 at the very beginning*/
    
    if(error0 < atol) return; /*already converged, directly return*/

    if(std::isnan(error)) {
        printf("Fatal error: GMRES norm inf\n");
        gmres_ctx->convrson = GMRES_DIV;
        return;
    }

    /*Normalize res = b - Ax_0 and copy it to first col of Q*/
    /*beta vector is the one appear in the least square problem*/
    /*e1 is not to be changed always as [1, 0, 0, 0, ...]^T*/
    if constexpr (std::is_same<float, T>::value) {
        float rnormi = 1.0f/rnorm;
        cublasScopy(blas_ctx->handle, xdim, res, 1, Q, 1);
        cublasSscal(blas_ctx->handle, xdim, &rnormi, Q, 1);
        // beta vector 
        cublasScopy(blas_ctx->handle, xdim, e1, 1, beta, 1);
        cublasSscal(blas_ctx->handle, xdim, &rnorm, beta, 1);
    } else
    if constexpr (std::is_same<double, T>::value){
        double rnormi = 1.0/rnorm;
        cublasDcopy(blas_ctx->handle, xdim, res,1, Q, 1);
        cublasDscal(blas_ctx->handle, xdim, &rnormi, Q, 1);
        // beta vector
        cublasDcopy(blas_ctx->handle, xdim, e1,1, beta, 1);
        cublasDscal(blas_ctx->handle, xdim, &rnorm, beta, 1);
    }
    
    for(unsigned int k=1;k<kspace+1;k++) {
        /*perform matrxi vector product approximation here*/
        MatDotVec(solctx, (void* ) v, (void*) (Q+xdim*(k-1)), xdim); 
        /*here v is the vector to be preconditioned*/
        //preconditioner(solctx, v, xdim);
        for(unsigned int j=0;j<k;j++) {
            T* hjk = h+j+(k-1)*(kspace+1);
            T* Qj  = Q+xdim*j;

            if constexpr (std::is_same<float, T>::value) {
                float htmp = 0.0f;
                // hjk = dot(v, Q)
                cublasSdot(blas_ctx->handle, xdim, v, 1, Qj, 1, &htmp);
                // copy htmp to hjk
                cudaMemcpy(hjk, &htmp, sizeof(float), cudaMemcpyHostToDevice);
                htmp *= -1.0f;
                // update v = v-hjk*Qj
                cublasSaxpy(blas_ctx->handle, xdim, &htmp, Qj, 1, v, 1);
            } else 
            if constexpr (std::is_same<double,T>::value) {
                double htmp = 0.0;
                // hjk = dot(v, Q)
                cublasDdot(blas_ctx->handle, xdim, v, 1, Qj, 1, &htmp);
                cudaMemcpy(hjk, &htmp, sizeof(double), cudaMemcpyHostToDevice);
                htmp *= -1.0;
                // update v = v-hjk*Qj
                cublasDaxpy(blas_ctx->handle, xdim, &htmp, Qj, 1, v, 1);
            }

        }
        // update h(k+1,k)
        T* hkp1k = h+k+(k-1)*(kspace+1);
        T vnorm;

        if constexpr (std::is_same<float, T>::value) {
            cublasSnrm2(blas_ctx->handle, xdim, v,   1, &vnorm);
        } else 
        if constexpr (std::is_same<double, T>::value) {
            cublasDnrm2(blas_ctx->handle, xdim, v,   1, &vnorm);
        }
        
        if(std::isnan(vnorm)) {
            gmres_ctx->convrson = GMRES_DIV;
            return;
        }

        T* Qkp1  = Q+xdim*k;

        // copy the value to vnorm
        cudaMemcpy(hkp1k, &vnorm, sizeof(T), cudaMemcpyHostToDevice);
        // normalize v
        if constexpr (std::is_same<float, T>::value) {
            float vnormi = 1.0f/vnorm;
            // update Q_k+1
            cublasScopy(blas_ctx->handle, xdim, v, 1, Qkp1, 1);
            cublasSscal(blas_ctx->handle, xdim, &vnormi, Qkp1, 1);
        } else 
        if constexpr (std::is_same<double, T>::value) {
            double vnormi = 1.0/vnorm;
            // update Q_k+1
            cublasDcopy(blas_ctx->handle, xdim, v, 1, Qkp1, 1);
            cublasDscal(blas_ctx->handle, xdim, &vnormi, Qkp1, 1);
        }

        /*update newest rotation matrix*/
        if constexpr (std::is_same<float, T>::value) {
            /*apply givens rotation for first k items*/
            /* input x <----- h0j*/
            /* input y <----- h1j*/
            apply_rot_f<<<1,1>>>(h+(k-1)*(kspace+1), cs, sn,k-1);
            /*apply givens totation to last two items*/
            givens_rot_f<<<1,1>>>(
                h+k-1+(k-1)*(kspace+1), h+k+0+(k-1)*(kspace+1), cs+k-1, sn+k-1
            );
        } else 
        if constexpr (std::is_same<double, T>::value) {
            /*apply givens rotation for first k items*/
            apply_rot_d<<<1,1>>>(h+(k-1)*(kspace+1), cs, sn,k-1);

            /*apply givens totation to last two items*/
            givens_rot_d<<<1,1>>>(
                h+k-1+(k-1)*(kspace+1), h+k+0+(k-1)*(kspace+1), cs+k-1, sn+k-1
            );

        }
        

        // rotate corresponding beta // since the last is 0
        // beta (k+1) = -sn(k)*beta(k)
        // beta (k  ) =  cs(k)*beta(k)
        if constexpr (std::is_same<double, T>::value) {
            dot_one_d<<<1,1>>>(sn+k-1,beta+k-1,beta+k,  -1.0);
            dot_one_d<<<1,1>>>(cs+k-1,beta+k-1,beta+k-1, 1.0);
        } else
        if constexpr (std::is_same<float, T>::value) {
            dot_one_f<<<1,1>>>(sn+k-1,beta+k-1,beta+k,  -1.0);
            dot_one_f<<<1,1>>>(cs+k-1,beta+k-1,beta+k-1, 1.0);
        }
        // error       = abs(beta(k + 1)) / b_norm;
        error = 0.0;
        cudaMemcpy(&error, beta+k, sizeof(T), cudaMemcpyDeviceToHost);

        error = std::fabs(error)/bnorm;
        
        cnt += 1;
        gmres_ctx->conv_iters += 1;
        if(error < atol) { 
            /*converged due to abs tol being satisfied*/
            gmres_ctx->convrson = GMRES_CONV_ABS;
            break;
        } else 
        if(error/error0 < rtol)  {
            /*converged due to rel tol being satisfied*/
            gmres_ctx->convrson = GMRES_CONV_REL;
            break;
        } else 
        if (gmres_ctx->conv_iters > gmres_ctx->maxiters) {
            /*converged due to rel tol being satisfied*/
            gmres_ctx->convrson = GMRES_CONV_MAX_ITER;
            break;
        }
    }

    if(gmres_ctx->convrson == GMRES_NOT_CONV) {
        restart = true;
    } else {
        restart = false;
    }

    /*calling solver to solve the triangular linear system*/
    if constexpr (std::is_same<float,  T>::value) {
        cublasStrsm(    
            blas_ctx->handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER, 
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            cnt,1,
            &P_1F,
            h, kspace+1,
            beta, cnt 
        );
    } else 
    if constexpr (std::is_same<double, T>::value) {
        cublasDtrsm(    
            blas_ctx->handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER, 
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            cnt,1,
            &P_1D,
            h, kspace+1,
            beta, cnt 
        );
    }

    /*after we solve the least square problem, we update x*/
    // update x = x+c*Qkp1
    if constexpr (std::is_same<float, T>::value) {
        for(unsigned int k=0;k<cnt;k++) {
            get_soln_f<<<1,256>>>(beta+k, x, Q+k*xdim, xdim);
        }
    } else 
    if constexpr (std::is_same<double, T>::value) {
        for(unsigned int k=0;k<cnt;k++) {
            get_soln_d<<<1,256>>>(beta+k, x, Q+k*xdim, xdim);
        }
    }
    
    /*since not converged */
    if (restart) {
        goto RESTART_ENTRY;
    }

    return;
}

#endif
