#include "gmres_ctx.cuh"
#include "cuda_constant.cuh"
#include <iostream>

void allocate_ram_gmres_app_ctx_d(
        double* &b,  double* &Q,  double* &h,  double* &v,
        double* &sn, double* &cs, double* &e1, double* &beta,
        unsigned int xdim, unsigned int kspace
){
    
    cudaMalloc((void **) &b,    sizeof(double)*xdim);
    cudaMalloc((void **) &Q,    sizeof(double)*xdim*(kspace+1));
    cudaMalloc((void **) &h,    sizeof(double)*kspace*(kspace+1));
    cudaMalloc((void **) &v,    sizeof(double)*xdim);

    cudaMalloc((void **) &sn,   sizeof(double)*(kspace+1));
    cudaMalloc((void **) &cs,   sizeof(double)*(kspace+1));
    cudaMalloc((void **) &e1,   sizeof(double)*(kspace+1));
    cudaMalloc((void **) &beta, sizeof(double)*(kspace+11));
}
    
void deallocate_ram_gmres_app_ctx_d(
        double* &b,  double* &Q,  double* &h,  double* &v,
        double* &sn, double* &cs, double* &e1, double* &beta
){
    cudaFree(b);
    cudaFree(Q);
    cudaFree(h);
    cudaFree(v);

    cudaFree(sn);
    cudaFree(cs);
    cudaFree(e1);
    cudaFree(beta);
}

void allocate_ram_gmres_app_ctx_f(
        float* &b,  float* &Q,  float* &h,  float* &v,
        float* &sn, float* &cs, float* &e1, float* &beta,
        unsigned int xdim, unsigned int kspace
){
    
    cudaMalloc((void **) &b,    sizeof(float )*xdim);
    cudaMalloc((void **) &Q,    sizeof(float )*xdim*(kspace+1));
    cudaMalloc((void **) &h,    sizeof(float )*kspace*(kspace+1));
    cudaMalloc((void **) &v,    sizeof(float )*xdim);

    cudaMalloc((void **) &sn,   sizeof(float )*(kspace+1));
    cudaMalloc((void **) &cs,   sizeof(float )*(kspace+1));
    cudaMalloc((void **) &e1,   sizeof(float )*(kspace+1));
    cudaMalloc((void **) &beta, sizeof(float )*(kspace+11));
}
    
void deallocate_ram_gmres_app_ctx_f(
        float* &b,  float* &Q,  float* &h,  float* &v,
        float* &sn, float* &cs, float* &e1, float* &beta
){
    cudaFree(b);
    cudaFree(Q);
    cudaFree(h);
    cudaFree(v);

    cudaFree(sn);
    cudaFree(cs);
    cudaFree(e1);
    cudaFree(beta);
}

void set_ones_const() {
    float  fno = -1.0;
    float  fpo =  1.0;
    double dno = -1.0;
    double dpo =  1.0;
    
    CUDA_CALL(cudaMemcpyToSymbol(P_ONE_F, &fpo, sizeof(float), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(N_ONE_F, &fno, sizeof(float), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(P_ONE_D, &dpo, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(N_ONE_D, &dno, sizeof(double), 0, cudaMemcpyHostToDevice));
    return;
}

__global__
void copy_to_b_d(double* bvec, double* gmresb,
                 unsigned int xdim,
                 unsigned int d, unsigned int* strides) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= xdim) return;
    gmresb[idx] = bvec[idx];
    printf("gmres[%d] %lf\n", idx, gmresb[idx]);
}

__global__
void copy_to_b_f(float* bvec, float* gmresb, 
                 unsigned int xdim,
                 unsigned int d, unsigned int* strides) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= xdim) return;
    gmresb[idx] = bvec[idx];
}

void set_b_vector_f(float* bvec, float* gmresb, 
                    unsigned int xdim,
                    unsigned int d, unsigned int* strides) {
    unsigned int threads_per_block = 256;
    unsigned int blocks = ceil(float(xdim)/threads_per_block);
    copy_to_b_f<<<blocks,threads_per_block>>>(bvec, gmresb, xdim, d, strides);   
}

void set_b_vector_d(double* bvec, double* gmresb, 
                    unsigned int xdim,
                    unsigned int d, unsigned int* strides) {
    printf("I am here\n");
    unsigned int threads_per_block = 256;
    unsigned int blocks = ceil(float(xdim)/threads_per_block);
    copy_to_b_d<<<blocks,threads_per_block>>>(bvec, gmresb, xdim, d, strides);   
}

