#ifndef __KRYIT_INTERFACE_H__
#define __KRYIT_INTERFACE_H__
#include "gmres.cuh"
#include "gmres_ctx.cuh"
#include "cublas_ctx.cuh"
#include "util.cuh"
extern "C" {
    // provide the data type to create the instance
    void* create_gmres_ctx(unsigned int size,
                           unsigned long int dim,
                           unsigned int etypes_in,
                           void* soasz_in,    // alignment
                           unsigned int iodim_in,
                           void* ioshape_in,  // ioshape
                           unsigned int datadim_in,
                           void* datashape_in,// actual data shape
                           unsigned int space,
                           void* atol, 
                           void* rtol); 
    void clean_gmres_ctx(void* input); 
    void* create_cublas_ctx(unsigned int size); 
    void clean_cublas_ctx(void* input); 
    void* gmres(unsigned int size);

    void gmres_solve(
                     void* matdotptr, // matrix-vec product approximation
                     void* gmreptr,   // gmres
                     void* solctx,    // solver application context
                     void* gctxptr,   // gmres application context
                     void* bctxptr    // cublas application context
                    );

    // some help function for debugging purpose
    void print_data(void* x, unsigned long int xdim, unsigned int dsize);
    void set_zero  (void* x, unsigned long int xdim, unsigned int dsize);
    void set_one   (void* x, unsigned long int xdim, unsigned int dsize);
    
    // function to set the address of the pmg src bank in PyFR
    void set_b_reg_addr(void* gctx, void* addr, unsigned int etype, unsigned int dsize);
    void set_curr_reg_addr(void* gctx, void* addr, unsigned int etype, unsigned int dsize);
    void check_b_reg_data(void* gctx, unsigned int ne, unsigned int len, unsigned int dsize);
    void check_curr_reg_data(void* gctx, unsigned int ne, unsigned int len, unsigned int dsize);
}

#endif



