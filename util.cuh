#ifndef __UTIL__CUH__
#define __UTIL__CUH__

#include <iostream>
#include "cuda_runtime.h"
#include <cmath>
#include <math.h>
/**
 * \fn utility function to print the data out for debugging purpose
 *
 */

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}


template<typename T>
__global__
void print_data(T* __restrict__ a, unsigned long int xdim) {
    unsigned long int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<xdim) printf("a[%ld] is %f\n", idx, a[idx]);
    return;
}


template<typename T> 
void print_data_wrapper(T* __restrict__ a, unsigned long int xdim) {
    unsigned int threads_per_block = 256;
    unsigned long int blocks = ceil(float(xdim)/threads_per_block);
    print_data<T> <<<blocks, threads_per_block>>>(a, xdim);
    cudaDeviceSynchronize();
}


void set_zeros_double(double*, unsigned long int xdim);
void set_zeros_float (float*,  unsigned long int xdim);
void set_ones_double (double*, unsigned long int xdim);
void set_ones_float  (float*,  unsigned long int xdim);

#endif
