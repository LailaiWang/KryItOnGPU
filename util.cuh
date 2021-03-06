#ifndef __UTIL__CUH__
#define __UTIL__CUH__

#include <iostream>
#include <vector>
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
std::vector<T> view_content(T* input, unsigned int cnts) {
	std::vector<T> output(cnts, 0);
	CUDA_CALL(cudaMemcpy(output.data(), input, sizeof(T) * cnts, cudaMemcpyDeviceToHost));
	return output;
}

template<typename T>
__global__
void print_data(T* __restrict__ a, unsigned long int xdim) {
    unsigned long int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<xdim) printf("a[%ld] is %f\n", idx, a[idx]);
    return;
}


template<typename T> 
void print_data_wrapper(T* a, unsigned long int xdim) {
    unsigned int threads_per_block = 256;
    unsigned long int blocks = ceil(float(xdim)/threads_per_block);
    print_data<T> <<<blocks, threads_per_block>>>(a, xdim);
    cudaDeviceSynchronize();
}

template<typename T>
__global__
void copy_array(T* dst, T* src, T scale, unsigned long int xdim) {
    unsigned long int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= xdim) return;
    dst[idx] = scale*src[idx];
}

template<typename T>
__global__
void copy_array(T* dst, T* src, T *scale, unsigned long int xdim) {
    unsigned long int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= xdim) return;
    dst[idx] = *scale*src[idx];
}

template<typename T>
__global__
void gpuSqrt(T* val) {
    *val = sqrt(*val);
}

template<typename T>
__global__
void accumulate_by_one(T* base, T* src) {
    *base += *src;
}

template<typename T>
__global__
void gpuReciprocal(T* val, T* valr) {
    *valr = (T)1.0/(*val);
}

template<typename T>
void copy_data_to_native(
          unsigned long long int* startingAddr, // input starting address
          unsigned long long int localAddr,  // local address
          unsigned int etype, // element type
          unsigned int datadim, // dimension of the datashape
          unsigned long int *datashapes, // datashape
          cudaStream_t stream
) {
    
    // Data on PyFR side for each element type are not necessarily contiguous
    // Data on native gmres side are contiguous
    T* local = (T*) (localAddr);

    unsigned long int esize[etype+1];
    esize[0] = 0;
    for(unsigned int ie=0;ie<etype;ie++) {
        esize[ie+1] = 1;
        for(unsigned int idm=0;idm<datadim;idm++) {
            esize[ie+1] *= datashapes[ie*datadim+idm];
        }
    }

    for(unsigned int ie=0;ie<etype;ie++) {
        T* estart = reinterpret_cast<T*> (startingAddr[ie]);
        unsigned long int offset = 0;
        for(unsigned int k=0;k<=ie;k++) {
            offset += esize[k];
        }

        unsigned long int nblocks = std::ceil((T) esize[ie+1]/256);   
        copy_array<<<nblocks,256,0,stream>>>(local+offset, estart, (T)1.0, esize[ie+1]);
    }

}



template<typename T>
void copy_data_to_user(
          unsigned long long int* startingAddr, // input starting address
          unsigned long long int localAddr,  // local address
          unsigned int etype, // element type
          unsigned int datadim, 
          unsigned long int * datashapes,// datashape
          cudaStream_t stream
){
    
    // cast to T* pointer
    T* local = (T*) (localAddr);
    unsigned long int esize[etype+1];
    esize[0] = 0;
    for(unsigned int ie=0;ie<etype;ie++) {
        esize[ie+1] = 1;
        for(unsigned int idm=0;idm<datadim;idm++) {
            esize[ie+1] *= datashapes[ie*datadim+idm];
        }
    }
    

    for(unsigned int ie=0;ie<etype;ie++) {
        T* estart = reinterpret_cast<T*> (startingAddr[ie]);
        unsigned long int offset = 0;
        for(unsigned int k=0;k<=ie;k++) {
            offset += esize[k];
        }
        unsigned long int nblocks = std::ceil((T) esize[ie+1]/256);   
        copy_array<<<nblocks,256,0,stream>>>(estart, local+offset, (T)1.0, esize[ie+1]);
    }
}


void set_zeros_double(double*, unsigned long int xdim, cudaStream_t);
void set_zeros_float (float*,  unsigned long int xdim, cudaStream_t);
void set_ones_double (double*, unsigned long int xdim, cudaStream_t);
void set_ones_float  (float*,  unsigned long int xdim, cudaStream_t);

#endif
