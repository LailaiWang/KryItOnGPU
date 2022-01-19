#ifndef GMRES_CTX_CUH
#define GMRES_CTX_CUH

#include "cuda_runtime.h"
#include "util.cuh"


void set_ones_const();

enum gmres_conv_reason {GMRES_NOT_CONV = 0,
                        GMRES_CONV_ABS = 1,
                        GMRES_CONV_REL = 2,
                        GMRES_CONV_MAX_ITER = 3,
                        GMRES_DIV=-1};


/*define some functions to manage to ram*/
void allocate_ram_gmres_app_ctx_d(
        double* &b,  double* &Q,  double* &h,  double* &v,
        double* &sn, double* &cs, double* &e1, double* &beta,
        unsigned int xdim, unsigned int kspace
); 
    
void deallocate_ram_gmres_app_ctx_d(
        double* &b,  double* &Q,  double* &h,  double* &v,
        double* &sn, double* &cs, double* &e1, double* &beta
);

void allocate_ram_gmres_app_ctx_f(
        float* &b,  float* &Q,  float* &h,  float* &v,
        float* &sn, float* &cs, float* &e1, float* &beta,
        unsigned int xdim, unsigned int kspace
); 
    
void deallocate_ram_gmres_app_ctx_f(
        float* &b,  float* &Q,  float* &h,  float* &v,
        float* &sn, float* &cs, float* &e1, float* &beta
); 
// definition of such will allow for future coupling with PyFR
void set_b_vector_f(float* bvec,  unsigned int bdim, unsigned int* bstrides);
void set_b_vector_d(double* bvec, unsigned int bdim, unsigned int* bstrides);

// passing pointer around
template<typename T> 
struct gmres_app_ctx {
    unsigned int xdim = 0;    // dimension of the square matrix
    unsigned int kspace = 0;  // dimension of the krylov subpspace max iteration number

    T atol=1e-11;
    T rtol=1e-10;

    gmres_conv_reason convrson = GMRES_NOT_CONV;
    
    unsigned int maxiters = 3500;
    unsigned int conv_iters = 0;

    gmres_app_ctx(unsigned int dim, unsigned int space, T at, T rt,
                 void (*allocate) (
                    T* &, T* &, T* &, T* &,
                    T* &, T* &, T* &, T* &,
                    unsigned int,
                    unsigned int
                 ),
                 void (*deallocate) (
                    T* &, T* &, T* &, T* &,
                    T* &, T* &, T* &, T* &
                 ),
                 void (*set_bvec) (
                    T*, unsigned int, unsigned int*
                 )
    ) {
        xdim   = dim;
        kspace = space;
        atol   = at;
        rtol   = rt;

        allocate_ram   = allocate;
        deallocate_ram = deallocate;
        set_b_vector = set_bvec;
    };

    T* b; /* a copy of b values in linear system Ax=b */
    T* Q;
    T* h;
    T* v;

    T* sn;
    T* cs;
    T* e1;
    T* beta; /* beta vector last 10 values space for some intermediate calculations */
    
    unsigned int bdim = 1;
    unsigned int *bstrides;

    void (*allocate_ram) (
        T* &, T* &, T* &, T* &,
        T* &, T* &, T* &, T* &,
        unsigned int,
        unsigned int
    );

    void (*deallocate_ram) (
        T* &, T* &, T* &, T* &,
        T* &, T* &, T* &, T* &
    );
    
    void (*set_b_vector) (T* , unsigned int , unsigned int*);
};

template<typename T>
struct precon_app_ctx {
    T* x;    // x stores the solution of Ax = b
    T* res;  // a vector to store res = b - Ax_i
    unsigned int xdim;  
    /*these two from spatial solver*/
    precon_app_ctx(T *xin, T *resin, unsigned int uk) {
        x   = xin;
        res = resin;
        xdim = uk;
    }
};


#endif
