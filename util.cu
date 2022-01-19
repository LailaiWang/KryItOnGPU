#include "util.cuh"

__global__
void set_zero_double(double* x, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < xdim) x[idx] = 0.0f;
}

__global__
void set_zero_float(float* x, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < xdim) x[idx] = 0.0f;
}

__global__
void set_one_double(double* x, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < xdim) {
        x[idx] = 1.0f;
    }
}

__global__
void set_one_float(float* x, unsigned int xdim) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < xdim) {
        x[idx] = 1.0f;
    }
}

void set_zeros_double(double* x, unsigned int xdim) {
    unsigned int blocks = std::ceil(double(xdim)/256);
    set_zero_double<<<blocks,256>>>(x,xdim);
}

void set_zeros_float(float* x, unsigned int xdim) {
    unsigned int blocks = std::ceil(float(xdim)/256);
    set_zero_float<<<blocks,256>>>(x,xdim);
}

void set_ones_double(double* x, unsigned int xdim) {
    unsigned int blocks = std::ceil(double(xdim)/256);
    set_one_double<<<blocks,256>>>(x,xdim);
}

void set_ones_float(float* x, unsigned int xdim) {
    unsigned int blocks = std::ceil(float(xdim)/256);
    set_one_float<<<blocks,256>>>(x,xdim);
}
