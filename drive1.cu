#include "arnoldi.cuh"
#include "MatDotVec.cuh"
#include "gmres.cuh"
#include "stdio.h"
#include <vector>
#include "util.cuh"
#include <iostream>
#include "cuda_runtime.h"
#include "cublas_v2.h"

int driver1() {
    
    unsigned int xdim    = 6;
    unsigned int kspace  = 5;
    
    std::vector<double> bvec(xdim, 1.0);
    std::vector<double> Qvec(xdim*(kspace+1), 1.0);
    std::vector<double> hvec(kspace*(kspace+1), 1.0);
    std::vector<double> vvec(xdim, 1.0);

    double *b = bvec.data();
    double *Q = Qvec.data();
    double *h = hvec.data();
    double *v = vvec.data();

    double *b_d;
    double *Q_d;
    double *h_d;
    double *v_d;
    
    cudaMalloc((void**) &b_d, xdim*sizeof(double));
    cudaMalloc((void**) &Q_d, xdim*(kspace+1)*sizeof(double));
    cudaMalloc((void**) &h_d, kspace*(kspace+1)*sizeof(double));
    cudaMalloc((void**) &v_d, xdim*sizeof(double));

    for (int i=0;i<xdim;i++) {
        b[i] = 1.0;
    }

    cudaMemcpy(b_d, b, xdim*sizeof(double), cudaMemcpyHostToDevice);
    
    // initialize the function pointer
    void (*fn) (double*, double*, unsigned int) = &MatDotVec_wrapper<double>;
    arnoldi<double>(fn, b_d, Q_d, h_d, v_d, xdim, kspace);
    
    cudaMemcpy(Q, Q_d, xdim*(kspace+1)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h, h_d, kspace*(kspace+1)*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(b_d);
    cudaFree(Q_d);
    cudaFree(h_d);
    cudaFree(v_d);
    
    return 0;
}

/*
int driver2() {

    unsigned int xdim    = 200;
    unsigned int kspace  = 200;
    
    std::vector<double> bvec(xdim, 1.0);
    std::vector<double> Qvec(xdim*(kspace+1), 0.0);
    std::vector<double> hvec(kspace*(kspace+1), 0.0);
    std::vector<double> vvec(xdim, 0.0);

    double *b = bvec.data();
    double *Q = Qvec.data();
    double *h = hvec.data();
    double *v = vvec.data();

    struct gmres_app_ctx<double> gctx(
        xdim, kspace, 1e-10, 1e-9, 
        &allocate_ram_gmres_app_ctx_d,
        &deallocate_ram_gmres_app_ctx_d
    );
    
    gctx.allocate_ram(
        gctx.b,  gctx.Q,  gctx.h,  gctx.v, 
        gctx.sn, gctx.cs, gctx.e1, gctx.beta,
        gctx.xdim, 
        gctx.kspace
    );

    cudaMemcpy(gctx.b, b, xdim*sizeof(double), cudaMemcpyHostToDevice);

    double *x_d, *res_d;
    cudaMalloc((void**) &x_d,   xdim*sizeof(double));
    cudaMalloc((void**) &res_d, xdim*sizeof(double));
    struct precon_app_ctx<double> pctx(x_d, res_d, xdim);
    
    struct cublas_app_ctx bctx (&initialize_cublas, &finalize_cublas);
    bctx.create_cublas(&bctx.handle);
    
    void (*fn) (double*, double*, unsigned int) = &MatDotVec_wrapper<double>;
    void (*fprecon) (void*, double*) = &MFPreconditioner<double>;

    void (*gmresSol) (
        void (*) (double*, double*, unsigned int),
        void (*) (void*, double*),
        void*,
        void*,
        void*
    ) = &MFgmres<double>;

    gmresSol(fn, fprecon, (void *) &ptxaddr, (void *) &ctxaddr, (void *) &btxaddr);
    
    cudaMemcpy(Q, gctx.Q, xdim*(kspace+1)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h, gctx.h, kspace*(kspace+1)*sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> soln(xdim, 0.0);

    cudaMemcpy(soln.data(), x_d, xdim*sizeof(double), cudaMemcpyDeviceToHost);
    
    for(auto &d : soln) std::cout<<"sol "<<d<<std::endl;

    bctx.clean_cublas(&bctx.handle);

    gctx.deallocate_ram(
        gctx.b, gctx.Q, gctx.h, gctx.v, 
        gctx.sn, gctx.cs, gctx.e1, gctx.beta
    );
    
    cudaFree(x_d);
    cudaFree(res_d);
    return 0;
}
*/
