#include "gmres_ctx.cuh"
#include "cuda_constant.cuh"
#include "util.cuh"
#include <iostream>

void allocate_ram_gmres_app_ctx(
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
    
void deallocate_ram_gmres_app_ctx(
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
