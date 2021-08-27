/* GMRES solver on GPU with matrix-free implementation 
 * Matrix-free matrix-vector product
 * Matrix-free preconditioner
 */ 

#ifndef GMRES_CUH
#define GMRES_CUH

#include "arnoldi.cuh"
#include "gmres_ctx.cuh"

template<typename T>
__device__
void calc_rotation(const T v1, const T v2, T* cs, T* cn) {
    double t = std::sqrt(v1*v1+v2*v2);
    cs = v1/t;
    sn = v2/t;
}

template<typename T> 
__device__
void apply_rotation() {


}

__global__
void set_zero_double(float* x, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < xdim) x[i] = 0.0f;
}

__global__
void set_zero_float(double* x, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < xdim) x[i] = 0.0f;
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

template<typename T>
void gmres(void (*MatDotVec) (T*, T*, unsigned int), // matrix-vector product
           void (*preconditioner(void*)), // preconditioner
           void *ctx  // application context
          ) {

    // cast the void pointer into a application context pointer

    // get the initial vector 
    //r = b - A*x;
    // get the norm of b
    
    set_zero_wrapper(sn, kspace);
    set_zero_wrapper(cs, kspace);
    set_zero_wrapper(e1, kspace+1);
    for(unsigned int k=1;k<m;k++) {
        // run arnoldi

        // eliminate the last element in H ith row and update the rotation matrix

        // update the residual vector


        // save the error

        if() {
            break;
        }
    }
    // calculate the result

}

#endif
