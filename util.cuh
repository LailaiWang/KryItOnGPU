#ifndef __UTIL__CUH__
#define __UTIL__CUH__

#include "cuda_runtime.h"
/**
 * \fn utility function to print the data out for debugging purpose
 *
 */

template<typename T>
__global__
void print_data(T* a, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<xdim) printf("a[%d] is %f\n", idx, a[idx]);
    return;
}


template<typename T> 
void print_data_wrapper(T* a, unsigned int xdim) {
    const unsigned int threads_per_block = 256;
    unsigned int blocks = ceil(float(xdim)/threads_per_block);
    print_data<T> <<<blocks, threads_per_block>>>(a, xdim);
    cudaDeviceSynchronize();
}


#endif
