#ifndef __KRYIT_INTERFACE_H__
#define __KRYIT_INTERFACE_H__
#include "gmres.cuh"
#include "gmres_ctx.cuh"
#include "cublas_ctx.cuh"
#include "MatDotVec.cuh"
#include "util.cuh"
extern "C" {
    // provide the data type to create the instance
    void* create_gmres_ctx(unsigned int size,
                           unsigned int dim,
                           unsigned int etypes_in,
                           unsigned int* soasz_in,    // alignment
                           unsigned int iodim_in,
                           unsigned int* ioshape_in,  // ioshape
                           unsigned int datadim_in,
                           unsigned int* datashape_in,// actual data shape
                           unsigned int space,
                           void* atol, 
                           void* rtol); 
    void clean_gmres_ctx(void* input); 
    void* create_precon_ctx(unsigned int size, unsigned int dim, void* xin, void* resin);
    void clean_precon_ctx(void* input); 
    void* create_cublas_ctx(unsigned int size); 
    void clean_cublas_ctx(void* input); 
    void* gmres(unsigned int size);

    void gmres_solve(
                     void* matdotptr, // matrix-vec product approximation
                     void* preconptr, // preconditioner
                     void* gmreptr,   // gmres
                     void* solctx,    // solver application context
                     void* pctxptr,   // preconditioner application context
                     void* gctxptr,   // gmres application context
                     void* bctxptr    // cublas application context
                    );

    void get_matdot(void* c, void* b, unsigned int xdim, unsigned int size);
    
    // some help function for debugging purpose
    void print_data(void* x, unsigned int xdim, unsigned int dsize);
    void set_zero  (void* x, unsigned int xdim, unsigned int dsize);
    void set_one   (void* x, unsigned int xdim, unsigned int dsize);

    void set_gmres_b_vector(void* gctx, void* data, unsigned int size); 
}

#endif



