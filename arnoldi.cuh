#ifndef ARNOLDI_CUH
#define ARNOLDI_CUH

#include <type_traits>
#include "mpi.h"
#include "cuda_runtime.h"
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include "cublas_v2.h"

template<unsigned int threads_per_block, typename T>
__global__ 
void vec_dot(T *vec_a, T *vec_b, T* norm, unsigned int n) {
    
    __shared__ T temp[threads_per_block];
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(idx >= n) return;

    temp[threadIdx.x] = vec_a[idx] * vec_b[idx];
    
    __syncthreads();
    
    if(threadIdx.x == 0) {
        T sum = 0.0;
        for (unsigned int i=0;i<threads_per_block;i++) {
            sum += temp[i];
        }
        atomicAdd(norm, sum);
    }
    return;
}

template<typename T>
__global__
void vec_normalize(T* vec, T* vecn, T *norm, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(idx<xdim) vecn[idx] = vec[idx]/std::sqrt(*norm);
}


template<typename T>
__global__
void assign_column(T* Q, T* q, unsigned int col, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // Q must be col major
    if(idx<xdim) {
        Q[idx+col*xdim] = q[idx];
    }
}

/** \fn function to compute the dot product of two vectors
 *  \param[in] input vector
 *  \param[in] input vector
 *  \param[out] output norm
 *  \param[in] dimension of the problem
 */
template<unsigned int threads_per_block, typename T> 
void vec_dot_wrapper(T* vec,
                     T* vecn,
                     T* vnorm,
                     unsigned int xdim) {
    unsigned int blocks = ceil(float(xdim)/threads_per_block);
    // do vec_dot for current PROCESS
    vec_dot<threads_per_block, T> <<< blocks, threads_per_block>>>(vec, vecn, vnorm, xdim);
    // if define MPI need to sum them
#ifdef _MPI
    T* norm;
    cudaMalloc((void **) &norm, sizeof(T));
    MPI_datatype dtype = MPI_DATATYPE_NULL;
    if constexpr (std::is_same<float, T>::value) {
        dtype = MPI_FLOAT;
    } else 
    if constexpr (std::is_same<double, T>::value){
        dtype = MPI_DOUBLE;
    } 
    // sum value into bnorm[1]
    MPI_Allreduce(vnorm, norm, 1, dtype, MPI_SUM, MPI_COMM_WORLD);
    // copy data to bnorm[0]
    cudaMemcpy(vnorm, norm, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaFree(norm);
#endif
}



/** \fn normalize a vector
 *  \param[in] vec vector to be normalized
 *  \param[out] out put of the normalized vector
 *  \param[int] dimension of the problem
 */
template<unsigned int threads_per_block, typename T> 
void vec_norm_wrapper(T* vec,
                      T* vecn,
                      unsigned int xdim) {
    unsigned int blocks = ceil(float(xdim)/threads_per_block);
    T* norm;
    cudaMalloc((void **) &norm, sizeof(T));
    vec_dot_wrapper <threads_per_block, T> (vec, vec, norm, xdim);
    vec_normalize<T><<< blocks, threads_per_block>>>(vec, vecn, norm, xdim);
    cudaFree(norm);
}

/** \fn normalize a vector and store in 
 *  \param[in] vec vector to be normalized
 *  \param[out] out put of the normalized vector
 *  \param[int] dimension of the problem
 */
template<unsigned int threads_per_block, typename T> 
void vec_normq_wrapper(T* vec,
                       T* vecn,
                       unsigned int col,
                       unsigned int xdim) {
    unsigned int blocks = ceil(float(xdim)/threads_per_block);
    T* norm;
    cudaMalloc((void **) &norm, sizeof(T));
    vec_dot_wrapper <threads_per_block, T> (vec, vec, norm, xdim);
    vec_normalize<T><<< blocks, threads_per_block>>>(vec, vecn, norm, xdim);
    cudaFree(norm);
}

template<unsigned int threads_per_block, typename T>
void update_hjk(T* Qj,  T* v, T* hjk, unsigned int xdim) {
    unsigned int blocks = ceil(float(xdim)/threads_per_block);
    vec_dot_wrapper<threads_per_block, T>(Qj, v, hjk, xdim);
    return;
}

template<typename T>
__global__
void update_v(T* v, T* hjk, T* Qj, unsigned int xdim) {
    
    // load hjk to shared memory
    __shared__ T hjk_;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    hjk_ = *hjk;
    __syncthreads();

    if(idx<xdim) {
        v[idx] = v[idx] - hjk_* Qj[idx];
    }
}

template<typename T>
__global__
void sqrt_point_value(T* v) {
    *v = std::sqrt(*v); 
    return;
}

/** \fn arnoldi iteration
 *  \param[in] MatDocVec helper function to do matrix vector product
 *  \param[in] b rhs vector dimension(xdim,1) b is not allowed to be changed
 *  \param[in] q dimesion (xdim, 1) 
 *  \param[out] Q dimesion (xdim,kspace+1) colum are orthonormal basis of krylov space
 *  \param[out] h dimension (kspace+1,kspace) upper hessenberg
 *  \param[in] v xdim
 *  \param[in] xdim dimension of the problem
 *  \param[in] dimension of the krylov space
 */
// place col in Q and h continuously as the Krylov vectors
// h(j,k) k 
// Q(:,j)


template<typename T>
void arnoldi(void (*MatDotVec)(T*, T*, unsigned int), 
             T* b,
             T* Q,
             T* h,
             T* v,
             unsigned int xdim,
             unsigned int kspace) {
    
    const unsigned int threads_per_block = 256;
    unsigned int blocks = ceil(float(xdim)/threads_per_block);
    
    // starting address of Q and h
    unsigned long long int Q0 = reinterpret_cast<unsigned long long int> (Q);
    unsigned long long int h0 = reinterpret_cast<unsigned long long int> (h);
    // normalize and store in the first col of Q
    vec_norm_wrapper<threads_per_block, T> (b, Q, xdim);

    for(unsigned int k=1;k<kspace+1;k++) {
        // calculate v as dot(A,q)
        // using matrix-free approximation
        MatDotVec(v, reinterpret_cast<T*>(Q0+sizeof(T)*xdim*(k-1)), xdim); 

        for(unsigned int j=0;j<k;j++) {
            // since cols of the matrix are stored continuously
            // the offset of hjk is k*(kspace+1) + j (h is a (kspace+1)*kspace matrix)
            T* hjk = reinterpret_cast<T*> (h0+sizeof(T)*(j+(k-1)*(kspace+1)));
            // the offset of Qj is  since Qj is a xdim*kspace matrix
            T* Qj  = reinterpret_cast<T*> (Q0+sizeof(T)*(xdim*j));
            update_hjk<threads_per_block, T> (Qj, v, hjk, xdim);
            // update v
            update_v  <T> <<<blocks, threads_per_block>>> (v, hjk, Qj, xdim);
        }
        
        T* hkp1k = reinterpret_cast<T*> (h0+sizeof(T)*((k)+(k-1)*(kspace+1)));
        vec_dot_wrapper<threads_per_block, T>(v, v, hkp1k, xdim);
        
        T norm_h = 0.0;
        cudaMemcpy(&norm_h, hkp1k, sizeof(T), cudaMemcpyDeviceToHost);

        if(norm_h>1e-12) {
            vec_normalize<T><<< blocks, threads_per_block >>>(
                v, reinterpret_cast<T*> (Q0+sizeof(T)*((k)*xdim)), hkp1k, xdim
            );
            sqrt_point_value<<<1,1>>>(hkp1k);
        } else {
            sqrt_point_value<<<1,1>>>(hkp1k);
            return;
        }

    }
    // reaching the max dimension
    return;
}
#endif
