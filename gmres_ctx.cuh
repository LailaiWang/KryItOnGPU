#ifndef GMRES_CTX_CUH
#define GMRES_CTX_CUH

#include "cuda_runtime.h"


void allocate_ram_gmres_app_ctx(
        double* &b,  double* &Q,  double* &h,  double* &v,
        double* &sn, double* &cs, double* &e1, double* &beta,
        unsigned int xdim, unsigned int kspace
); 
    
void deallocate_ram_gmres_app_ctx(
        double* &b,  double* &Q,  double* &h,  double* &v,
        double* &sn, double* &cs, double* &e1, double* &beta
); 

void set_ones_const();

enum gmres_conv_reason {GMRES_NOT_CONV = 0,
                        GMRES_CONV_ABS = 1,
                        GMRES_CONV_REL = 2,
                        GMRES_CONV_MAX_ITER = 3,
                        GMRES_DIV=-1};

// passing pointer around
struct gmres_app_ctx {
    unsigned int xdim = 0;    // dimension of the square matrix
    unsigned int kspace = 0;  // dimension of the krylov subpspace max iteration number

    double atol=1e-11;
    double rtol=1e-10;

    gmres_conv_reason convrson = GMRES_NOT_CONV;
    
    unsigned int maxiters = 3500;
    unsigned int conv_iters = 0;

    gmres_app_ctx(unsigned int dim, unsigned int space, double at, double rt,
                 void (*allocate) (
                    double* &, double* &, double* &, double* &,
                    double* &, double* &, double* &, double* &,
                    unsigned int,
                    unsigned int
                 ),
                 void (*deallocate) (
                    double* &, double* &, double* &, double* &,
                    double* &, double* &, double* &, double* &
                 )
    ) {
        xdim   = dim;
        kspace = space;
        atol   = at;
        rtol   = rt;

        allocate_ram   = allocate;
        deallocate_ram = deallocate; 
    };

    double* b; /* a copy of b values in linear system Ax=b */
    double* Q;
    double* h;
    double* v;

    double* sn;
    double* cs;
    double* e1;
    double* beta; /* beta vector last 10 values space for some intermediate calculations */

    
    void (*allocate_ram) (
        double* &, double* &, double* &, double* &,
        double* &, double* &, double* &, double* &,
        unsigned int,
        unsigned int
    );

    void (*deallocate_ram) (
        double* &, double* &, double* &, double* &,
        double* &, double* &, double* &, double* &
    );

};

struct precon_app_ctx {
    double* x;
    double* res;
    unsigned int xdim;  
    /*these two from spatial solver*/
    precon_app_ctx(double *xin, double *resin, unsigned int uk) {
        x   = xin;
        res = resin;
        xdim = uk;
    }
};


#endif
