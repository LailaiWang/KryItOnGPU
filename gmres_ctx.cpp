#include "gmres_ctx.cuh"

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
