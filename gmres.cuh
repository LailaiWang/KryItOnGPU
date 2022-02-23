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

#include "mpi.h"
#include "mpi-ext.h"
#if defined(MPIX_CUDA_AWARE_SUPPORT)
#include "mpi-ext.h"
#endif

template<typename T>
__global__
void givens_rot(T* __restrict__ h1, T* __restrict__ h2, T* __restrict__ c, T* __restrict__ s) {
    T tmp = sqrt((*h1)*(*h1)+(*h2)*(*h2));
    *c = *h1/tmp;
    *s = *h2/tmp;
    *h1 = (*c)*(*h1)+(*s)*(*h2);
    *h2 = 0.0f;
    return;
}

template<typename T>
__global__
void apply_rot(T* __restrict__ h, T* __restrict__ cs, T* __restrict__ sn, unsigned int k) {
    T temp = 0.0f;
    for(unsigned int i=0;i<k;i++) {
        temp = cs[i]*h[i]+sn[i]*h[i+1];
        h[i+1] = -sn[i]*h[i]+cs[i]*h[i+1];
        h[i  ] = temp;
    }
    return;
}

template<typename T>
__global__
void apply_rot_beta(T* __restrict__ beta, T* __restrict__ cs, T* __restrict__ sn, unsigned int k) {
    T temp = 0.0f;
    for(unsigned int i=0;i<k;i++) {
        temp = cs[i]*beta[i]+sn[i]*beta[i+1];
        beta[i+1] = -sn[i]*beta[i]+cs[i]*beta[i+1];
        beta[i  ] = temp;
    }
    return;
}

template<typename T>
__global__
void dot_one(T* __restrict__ a, T* __restrict__ b, T* __restrict__ c, T scal) {
    *c = (*a)*(*b)*scal;
    return;
}

template<typename T>
auto get_addr(unsigned long long int start, unsigned int offset) {

    return reinterpret_cast<T*> (start+sizeof(T)*offset);
}

template<typename T>
__global__
void get_soln(T* __restrict__ wts, T* __restrict__ x, T* __restrict__ q, unsigned long int xdim) {
    unsigned long int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= xdim) return;
    x[idx] += (*wts)*q[idx];
    return;
}

/** 
 * \fn a function to set array to zero on the device
 *
 */
template<typename T>
void set_zero_wrapper(T* x, unsigned long int xdim, cudaStream_t stream) {
    if constexpr (std::is_same<float, T>::value) {
        set_zeros_float(x, xdim, stream);
    } else
    if constexpr (std::is_same<double, T>::value){
        set_zeros_double(x, xdim, stream);
    }
}

/* \fn norm2 of input vector vec
 */
template<typename T> 
void mGPU_norm2_wrapper(cublasHandle_t handle,
                        unsigned long int xdim,
                        T* vec,
                        T* vnorm,
                        MPI_Comm comm) {
    T localnorm;
    if constexpr (std::is_same<float,T>::value) {
        cublasSdot(handle, xdim, vec, 1, vec, 1, &localnorm);
    } else 
    if constexpr (std::is_same<double, T>:: value) {
        cublasDdot(handle, xdim, vec, 1, vec, 1, &localnorm);
    }

    MPI_Datatype dtype = MPI_DATATYPE_NULL;

    if constexpr (std::is_same<float, T>::value) {
        dtype = MPI_FLOAT;
    } else 
    if constexpr (std::is_same<double, T>::value){
        dtype = MPI_DOUBLE;
    } 
    // sum value into bnorm[1]
    MPI_Allreduce(&localnorm, vnorm, 1, dtype, MPI_SUM, comm);
    *vnorm = std::sqrt(*vnorm);
    return;
}

/* \fn dot product of input vector avec and bvec
 */
template<typename T>
void mGPU_dot_wrapper(cublasHandle_t handle, 
                      unsigned long int xdim,
                      T* avec, T* bvec, 
                      T* vdot,
                      MPI_Comm comm) {
    T localnorm;
    if constexpr (std::is_same<float,T>::value) {
        cublasSdot(handle, xdim, avec, 1, bvec, 1, &localnorm);
    } else 
    if constexpr (std::is_same<double, T>:: value) {
        cublasDdot(handle, xdim, avec, 1, bvec, 1, &localnorm);
    }

    MPI_Datatype dtype = MPI_DATATYPE_NULL;

    if constexpr (std::is_same<float, T>::value) {
        dtype = MPI_FLOAT;
    } else 
    if constexpr (std::is_same<double, T>::value){
        dtype = MPI_DOUBLE;
    } 
    MPI_Allreduce(&localnorm, vdot, 1, dtype, MPI_SUM, comm);
    return;
}

/*wrapper to compute the norm of discontinuous data on user side*/
template<typename T>
void mGPU_dot_breg_wrapper(void* gctx, T* dotval, cublasHandle_t handle) {
    struct gmres_app_ctx<T>* gmres_ctx = (struct gmres_app_ctx<T>*) (gctx);
    unsigned int etype = gmres_ctx->etypes;
    unsigned int datadim = gmres_ctx->datadim;
    T sum = 0;
    for(unsigned int ie=0;ie<etype;ie++) {
        unsigned long int dimpertype = 1;
        for(unsigned int id=0;id<datadim;id++) {
            dimpertype *= gmres_ctx->datashape[ie*datadim+id];
        }
        
        T* b = reinterpret_cast<T*> (gmres_ctx->b_reg[ie]);
        
        T localnorm = 0;
        if constexpr (std::is_same<float,T>::value) {
            cublasSdot(handle, dimpertype, b, 1, b, 1, &localnorm);
        } else 
        if constexpr (std::is_same<double, T>:: value) {
            cublasDdot(handle, dimpertype, b, 1, b, 1, &localnorm);
        }

        sum += localnorm;
    }

    MPI_Datatype dtype = MPI_DATATYPE_NULL;
    if constexpr (std::is_same<float, T>::value) {
        dtype = MPI_FLOAT;
    } else 
    if constexpr (std::is_same<double, T>::value){
        dtype = MPI_DOUBLE;
    } 
    MPI_Allreduce(&sum, dotval, 1, dtype, MPI_SUM, gmres_ctx->mpicomm);
    return;

}

/*wrapper to compute the norm of discontinuous data on user side*/
template<typename T>
void mGPU_dot_creg_wrapper(void* gctx, T* dotval, cublasHandle_t handle) {
    struct gmres_app_ctx<T>* gmres_ctx = (struct gmres_app_ctx<T>*) (gctx);
    unsigned int etype = gmres_ctx->etypes;
    unsigned int datadim = gmres_ctx->datadim;
    T sum = 0;
    for(unsigned int ie=0;ie<etype;ie++) {
        unsigned long int dimpertype = 1;
        for(unsigned int id=0;id<datadim;id++) {
            dimpertype *= gmres_ctx->datashape[ie*datadim+id];
        }
        
        T* b = reinterpret_cast<T*> (gmres_ctx->curr_reg[ie]);
        
        T localnorm = 0;
        if constexpr (std::is_same<float,T>::value) {
            cublasSdot(handle, dimpertype, b, 1, b, 1, &localnorm);
        } else 
        if constexpr (std::is_same<double, T>:: value) {
            cublasDdot(handle, dimpertype, b, 1, b, 1, &localnorm);
        }

        sum += localnorm;
    }

    MPI_Datatype dtype = MPI_DATATYPE_NULL;

    if constexpr (std::is_same<float, T>::value) {
        dtype = MPI_FLOAT;
    } else 
    if constexpr (std::is_same<double, T>::value){
        dtype = MPI_DOUBLE;
    } 
    MPI_Allreduce(&sum, dotval, 1, dtype, MPI_SUM, gmres_ctx->mpicomm);
    return;
}

/** 
 * \fn gmres function
 * MatDotVec is from PyFR, we use a right preconditioning technique.
 * Right preconditioning does not change the norm we are  trying to minimise.
 */
template<typename T>
void MFgmres(
      void (*MatDotVec) (void*, bool), /* matrix-vector product func*/
      void* solctx, /*solver context defined as provided*/
      void* gtx,
      void* btx,
      unsigned int icnt
    ) {
    /* grab different application context*/
    struct gmres_app_ctx<T>* gmres_ctx = (struct gmres_app_ctx<T>*) (gtx);
    struct cublas_app_ctx* blas_ctx = (struct cublas_app_ctx*) (btx);

    /* dimension of the problem and krylov subspace*/
    unsigned long int xdim = gmres_ctx->xdim;
    unsigned int kspace = gmres_ctx->kspace;
    
    /*small vectors*/
    T* sn   = gmres_ctx->sn;    // dimension kspace+1
    T* cs   = gmres_ctx->cs;    // dimension kspace+1
    T* beta = gmres_ctx->beta;  // dimension kspace+11
    
    /*large vectors*/
    T* Q = gmres_ctx->Q; // dimension xdim*(kspace+1)
    T* h = gmres_ctx->h; // dimension (kspace+1)*kspace
    T* v = gmres_ctx->v; // dimension  xdim

    T atol = gmres_ctx->atol;
    T rtol = gmres_ctx->rtol;
    
    unsigned long int nblocks = std::ceil((T)xdim/256);

    bool restart = false;  /*if it is restart or not*/

    unsigned int cnt = 0;  /*reset when restart, dimension of the least square problem*/

    gmres_ctx->conv_iters = 0;  /**/

    T error  = 0.0; /*residual*/
    T error0 = 0.0; 

    T bnorm, rnorm;
    
    /*compute the norm of b vector, we do not store b in gmres ctx, b is on PyFR*/
    mGPU_dot_breg_wrapper<T>(gtx, &bnorm, blas_ctx->handle);
    bnorm = std::sqrt(bnorm);

    /*set up the initial value*/
    set_zero_wrapper(v,   xdim, blas_ctx->stream);   

    bool init = true;

    RESTART_ENTRY: {  /*restart is essentially using a better guess*/
        if(cnt != 0 ) init = false;
        cnt = 0;  /*no. of iterations till convergence need to reset when restart*/
    }
    
    set_zero_wrapper(sn,   kspace+1, blas_ctx->stream);   /*initialization to 0*/
    set_zero_wrapper(cs,   kspace+1, blas_ctx->stream);
    set_zero_wrapper(beta, kspace+11,blas_ctx->stream);

    /*copy the initial guess to the current reg bank in PyFR*/
    gmres_ctx->copy_to_user(
        gmres_ctx->x_reg, reinterpret_cast<unsigned long long int> (v), 
        gmres_ctx->etypes, gmres_ctx->datadim, gmres_ctx->datashape, blas_ctx->stream
    );

    MatDotVec(solctx, init);

    gmres_ctx->copy_to_native(
        gmres_ctx->curr_reg, reinterpret_cast<unsigned long long int> (Q), 
        gmres_ctx->etypes, gmres_ctx->datadim, gmres_ctx->datashape, blas_ctx->stream
    );
    
    /* compute the norm of r = b - Ax */
    /* for initial guess x = 0, for restart x != 0*/
    mGPU_norm2_wrapper<T>(blas_ctx->handle, xdim, Q, &rnorm, gmres_ctx->mpicomm);
    
    error = rnorm/bnorm;

    if (restart == false ) error0 = error; /*only update error0 at the very beginning*/
    
    if(error0 < atol) return; /*already converged, directly return*/

    if(std::isnan(error)) {
        printf("Fatal error: GMRES norm inf\n");
        gmres_ctx->convrson = GMRES_DIV;
        return;
    }

    /*setting up initial beta vector*/
    cudaMemcpy(beta, &rnorm, sizeof(T), cudaMemcpyHostToDevice);

    /*normalize Q*/
    if constexpr (std::is_same<float, T>::value) {
        float rnormi = 1.0f/rnorm;
        cublasSscal(blas_ctx->handle, xdim, &rnormi, Q, 1);
    } else
    if constexpr (std::is_same<double, T>::value){
        double rnormi = 1.0/rnorm;
        cublasDscal(blas_ctx->handle, xdim, &rnormi, Q, 1);
    }
    
    for(unsigned int k=1;k<kspace+1;k++) {
        /*perform matrix vector product approximation here*/
        /*v = Aq, need to copy q into the current bank in PyFR*/
        gmres_ctx->copy_to_user(
            gmres_ctx->curr_reg, reinterpret_cast<unsigned long long int> (Q+(k-1)*xdim), 
            gmres_ctx->etypes, gmres_ctx->datadim, gmres_ctx->datashape, blas_ctx->stream
        );

        MatDotVec(solctx, false);
        
        /*v = Aq is obtained in PyFR, need to copy to v*/
        gmres_ctx->copy_to_native(
            gmres_ctx->curr_reg, reinterpret_cast<unsigned long long int> (v),
            gmres_ctx->etypes, gmres_ctx->datadim, gmres_ctx->datashape, blas_ctx->stream
        );

        for(unsigned int j=0;j<k;j++) {
            T* hjk = h+j+(k-1)*(kspace+1);
            T* Qj  = Q+xdim*j;
            T htmp = 0.0;

            mGPU_dot_wrapper<T>(blas_ctx->handle, xdim, v, Qj, &htmp, gmres_ctx->mpicomm);

            cudaMemcpy(hjk, &htmp, sizeof(T), cudaMemcpyHostToDevice);
            
            htmp *= -1.0;

            if constexpr (std::is_same<float, T>::value) {
                /* update v = v-hjk*Qj */
                cublasSaxpy(blas_ctx->handle, xdim, &htmp, Qj, 1, v, 1);
            } else 
            if constexpr (std::is_same<double,T>::value) {
                /* update v = v-hjk*Qj */
                cublasDaxpy(blas_ctx->handle, xdim, &htmp, Qj, 1, v, 1);
            }

        }
        /* update h(k+1,k) */
        T* hkp1k = h+k+(k-1)*(kspace+1);
        T vnorm;
        
        mGPU_norm2_wrapper(blas_ctx->handle, xdim, v, &vnorm, gmres_ctx->mpicomm);

        if(std::isnan(vnorm)) {
            gmres_ctx->convrson = GMRES_DIV;
            return;
        }

        T* Qkp1  = Q+xdim*k;

        // copy the value to vnorm
        cudaMemcpy(hkp1k, &vnorm, sizeof(T), cudaMemcpyHostToDevice);
        
        /*normalize v*/
        copy_array<<<nblocks,256,0,blas_ctx->stream>>>(Qkp1, v, (T)1.0/vnorm, xdim);

        /*apply givens rotation for first k items*/
        /* input x <----- h0j*/
        /* input y <----- h1j*/
        apply_rot<T><<<1, 1, 0, blas_ctx->stream>>>(h+(k-1)*(kspace+1), cs, sn,k-1);
        /*apply givens totation to last two items*/
        givens_rot<T><<<1,1, 0, blas_ctx->stream>>>(
                h+k-1+(k-1)*(kspace+1), h+k+0+(k-1)*(kspace+1), cs+k-1, sn+k-1
        );
        
        /* rotate corresponding beta  since the last is 0 */
        /* beta (k+1) = -sn(k)*beta(k) */
        /* beta (k  ) =  cs(k)*beta(k) */
        dot_one<T><<<1,1,0,blas_ctx->stream>>>(sn+k-1, beta+k-1, beta+k,  -1.0);
        dot_one<T><<<1,1,0,blas_ctx->stream>>>(cs+k-1, beta+k-1, beta+k-1, 1.0);
        error = 0.0;

        cudaMemcpy(&error, beta+k, sizeof(T), cudaMemcpyDeviceToHost);
        
        error = std::fabs(error)/bnorm;
        
        cnt += 1;
        gmres_ctx->conv_iters += 1;

        /*for the first pseudo iteration in PTC, force to continue till reach maximum dimension*/
        if (icnt != 0 ) {
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
    
    /*get the solution*/
    set_zero_wrapper(v,   xdim, blas_ctx->stream);   
    for(unsigned int k=0;k<cnt;k++) {
        unsigned long int blocks = std::ceil((T) xdim/256);
        get_soln<<<blocks,256,0,blas_ctx->stream>>>(beta+k, v, Q+k*xdim, xdim);
    }
    /*since not converged */
    if (restart && icnt != 0) {
        /*only allow restart except the first iteration in PTC*/
        goto RESTART_ENTRY;
    }

    /* solution is stored in x */
    /* copy solution for native data layout to PyFR data layout*/
    gmres_ctx->copy_to_user(
       gmres_ctx->x_reg, reinterpret_cast<unsigned long long int> (v), 
       gmres_ctx->etypes, gmres_ctx->datadim, gmres_ctx->datashape, blas_ctx->stream
    );
    
    /*For the first pseudo iteration, backup the krylov space*/
    if(icnt == 0) {
        T* Qf  = gmres_ctx->Qf;
        T* hf  = gmres_ctx->hf;
        T* snf = gmres_ctx->snf;
        T* csf = gmres_ctx->csf;
        nblocks = std::ceil((T)xdim*(kspace+1)/256);
        copy_array<<<nblocks, 256, 0, blas_ctx->stream>>>(Qf,  Q,  (T)1.0, xdim*(kspace+1));
        nblocks = std::ceil((T) kspace*(kspace+1)/256);
        copy_array<<<nblocks, 256, 0, blas_ctx->stream>>>(hf,  h,  (T)1.0, kspace*(kspace+1));
        nblocks = std::ceil((T)(kspace+1)/256);
        copy_array<<<nblocks, 256, 0, blas_ctx->stream>>>(snf, sn, (T)1.0, kspace+1);
        copy_array<<<nblocks, 256, 0, blas_ctx->stream>>>(csf, cs, (T)1.0, kspace+1);
        /*no need to copy vf and betaf*/
    }
    /*synchronize our stream before we return back to PyFR*/
    cudaStreamSynchronize(blas_ctx->stream);
    return;
}

/*
 * \fn function for the preconditioner with freezed krylov subspaces
 * (I/Δt+I/Δτ-a_ii ∂ R/ ∂ q) Y = X
 * (I/Δt + I/Δτ + I/Δτ* - a_ii ∂ R/ ∂ q)Y = R - (I/Δt + I/Δτ ) q +
 *                                          X - a_ii R_0 + (I/Δt + I/Δτ ) q 
 * Let Δτ* = 0,
 * PY = R - (I/Δt + I/Δτ ) q + SRC(X) = b*
 * If just one iteration PY = b*
 * Consider frozen Kylov subspace vectors [v_0, v_1, ..., v_k]
 * We can use these frozen vectors, apply givens rotation to new || b* ||e_1
 * Resolve the triangluar system
 */
template<typename T>
void  PreconditioningWithFrozenKrylov(void* gtx, void* btx) {

    struct gmres_app_ctx<T>* gmres_ctx = (struct gmres_app_ctx<T>*) (gtx);
    struct cublas_app_ctx* blas_ctx = (struct cublas_app_ctx*) (btx);
    
    unsigned long int xdim = gmres_ctx->xdim;
    unsigned int kspace = gmres_ctx->kspace;

    T* sn   = gmres_ctx->snf;
    T* cs   = gmres_ctx->csf;
    T* beta = gmres_ctx->betaf;

    T* Q = gmres_ctx->Qf;
    T* h = gmres_ctx->hf;
    T* v = gmres_ctx->vf;

    /*evaluate the norm*/
    T vnorm;
    mGPU_dot_creg_wrapper(gtx, &vnorm, blas_ctx->handle);
    vnorm = std::sqrt(vnorm);
    /*copy this value to beta*/
    cudaMemcpy(beta, &vnorm, sizeof(T), cudaMemcpyHostToDevice);
    /*rotate beta*/


    /*resolve the linear system*/
    /*calling solver to solve the triangular linear system*/
    if constexpr (std::is_same<float,  T>::value) {
        cublasStrsm(    
            blas_ctx->handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER, 
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            kspace,1,
            &P_1F,
            h, kspace+1,
            beta,  kspace
        );
    } else 
    if constexpr (std::is_same<double, T>::value) {
        cublasDtrsm(    
            blas_ctx->handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER, 
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            kspace,1,
            &P_1D,
            h, kspace+1,
            beta, kspace 
        );
    }

    /*get the solution*/
    set_zero_wrapper(v,   xdim, blas_ctx->stream);   
    unsigned long int blocks = std::ceil((T) xdim/256);
    for(unsigned int k=0;k<kspace;k++) {
        get_soln<<<blocks,256,0,blas_ctx->stream>>>(beta+k, v, Q+k*xdim, xdim);
    }

    /*synchronize our stream before we return back to PyFR*/
    cudaStreamSynchronize(blas_ctx->stream);
    return;
}

#endif
