#ifndef GMRES_CTX_CUH
#define GMRES_CTX_CUH

#include "cuda_runtime.h"
#include "util.cuh"

#include "mpi.h"
#include "mpi-ext.h"
#if defined(MPIX_CUDA_AWARE_SUPPORT)
#include "mpi-ext.h"
#endif

template<typename T>
void allocate_ram_gmres_app_ctx(
        T* &Q,  T* &h,  T* &v,
        T* &sn, T* &cs, T* &e1, T* &beta,
        bool* &nonpadding,
        unsigned long int xdim, unsigned int kspace
){
    
    cudaMalloc((void **) &Q,    sizeof(T)*xdim*(kspace+1));
    cudaMalloc((void **) &h,    sizeof(T)*kspace*(kspace+1));
    cudaMalloc((void **) &v,    sizeof(T)*xdim);

    cudaMalloc((void **) &sn,   sizeof(T)*(kspace+1));
    cudaMalloc((void **) &cs,   sizeof(T)*(kspace+1));
    cudaMalloc((void **) &e1,   sizeof(T)*(kspace+1));
    cudaMalloc((void **) &beta, sizeof(T)*(kspace+11));
    cudaMalloc((void **) &nonpadding, sizeof(bool)*xdim);
}

template<typename T>
void deallocate_ram_gmres_app_ctx(
        T* &Q,  T* &h,  T* &v,
        T* &sn, T* &cs, T* &e1, T* &beta, 
        bool* &nonpadding
){
    cudaFree(Q);
    cudaFree(h);
    cudaFree(v);

    cudaFree(sn);
    cudaFree(cs);
    cudaFree(e1);
    cudaFree(beta);
    cudaFree(nonpadding);
}

void set_ones_const();
enum gmres_conv_reason {GMRES_NOT_CONV = 0,
                        GMRES_CONV_ABS = 1,
                        GMRES_CONV_REL = 2,
                        GMRES_CONV_MAX_ITER = 3,
                        GMRES_DIV=-1};

void set_reg_addr_pyfr(
    unsigned long long int*, // input
    unsigned long long int*, // local
    unsigned int // etype
);

void fill_nonpadding(unsigned int etype, 
                     unsigned long int* soasz,
                     unsigned int iodim,
                     unsigned long int* ioshape,
                     unsigned int datadim,
                     unsigned long int* datashape,
                     unsigned long int xdim,
                     bool* nonpadding);

// passing pointer around
template<typename T> 
struct gmres_app_ctx {
    
    MPI_Comm mpicomm;
    unsigned int nranks;

    unsigned long int xdim = 0;    // dimension of the square matrix
    unsigned int kspace = 0;  // dimension of the krylov subpspace max iteration number
    
    T atol=1e-11;
    T rtol=1e-10;
    
    gmres_conv_reason convrson = GMRES_NOT_CONV;
    unsigned int maxiters = 3500;
    unsigned int conv_iters = 0;

    unsigned int etypes; // maximum tri quad hex tet prism pyrimid
    unsigned long int soasz[2];
    unsigned int iodim;
    unsigned long int ioshape[200]; // [nsp, nvar, neles]
    unsigned int datadim;
    unsigned long int datashape[200]; // [nblocks, nsp, neles/nblocks/soasz, nvar, soasz]
    
    // define an array of bool on device

    bool* nonpadding; // add a padding value for gmres calculation, skip padding space 

    unsigned long long int b_reg[10];    // starting address for b vector on PyFR side
    unsigned long long int x_reg[10];    // x soln on PyFR side
    unsigned long long int curr_reg[10]; // starting address for current operating space on PyFR

    gmres_app_ctx(MPI_Comm mpicom_in,
                  unsigned int nranks_in,
                  unsigned long int dim,
                  unsigned int etypes_in,
                  unsigned long int* soasz_in,
                  unsigned int iodim_in,
                  unsigned long int* ioshape_in,
                  unsigned int datadim_in,
                  unsigned long int* datashape_in,
                  unsigned int space,
                  T at, T rt,
                  void (*allocate) (
                    T* &, T* &, T* &,
                    T* &, T* &, T* &, T* &,
                    bool* &,
                    unsigned long int,
                    unsigned int
                  ),
                  void (*deallocate) (
                    T* &, T* &, T* &,
                    T* &, T* &, T* &, T* &,
                    bool* &
                  ),
                  void (*fillpad) (
                    unsigned int, unsigned long int*,
                    unsigned int, unsigned long int*,
                    unsigned int, unsigned long int*,
                    unsigned long int, bool*
                  ),

                  void (*copy_to_nativevec)(
                    unsigned long long int*, // input starting address
                    unsigned long long int,  // local stress
                    unsigned int, // element type
                    unsigned int, // datadim
                    unsigned long int * // datashape
                  ),

                  void (*copy_to_uservec)(
                    unsigned long long int*, // input starting address
                    unsigned long long int,  // local stress
                    unsigned int, // element type
                    unsigned int, //datadim
                    unsigned long int * // datashape
                  ), 

                  void (*set_reg_addr_inp) (
                    unsigned long long int*, // input
                    unsigned long long int*, // native
                    unsigned int// etype
                  )

    ) {
        mpicomm = mpicom_in;
        nranks = nranks_in;
        xdim   = dim; // dimension of the problem // not continuous due to alignment
        etypes = etypes_in;
    
        for(unsigned int i=0;i<2;i++) {
            soasz[i] = soasz_in[i];
        }

        iodim = iodim_in;
        for(unsigned int i=0;i<iodim*etypes;i++) {
            ioshape[i] = ioshape_in[i];
        }

        datadim = datadim_in;
        for(unsigned int i=0;i<datadim*etypes;i++) {
            datashape[i] = datashape_in[i];
        }

        kspace = space;
        atol   = at;
        rtol   = rt;

        allocate_ram   = allocate;
        deallocate_ram = deallocate;

        fill = fillpad;
        copy_to_native = copy_to_nativevec;
        copy_to_user   = copy_to_uservec;
        set_reg_addr   = set_reg_addr_inp;
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
        T* &, T* &, T* &,
        T* &, T* &, T* &, T* &, bool* &,
        unsigned long int,
        unsigned int
    );

    void (*deallocate_ram) (
        T* &, T* &, T* &,
        T* &, T* &, T* &, T* &, bool* &
    );
    
    void (*fill) (unsigned int, unsigned long int*, 
                  unsigned int, unsigned long int*,
                  unsigned int, unsigned long int*,
                  unsigned long int, bool*);

    void (*copy_to_native)(
          unsigned long long int*, // input starting address
          unsigned long long int,  // local address
          unsigned int,            // element type
          unsigned int,            // datadim
          unsigned long int *      // datashape
    );
    void (*copy_to_user) (
          unsigned long long int*, // input starting address
          unsigned long long int,  // local address
          unsigned int,            // element type
          unsigned int,            // datadim
          unsigned long int *      // datashape
    );
    void (*set_reg_addr) (
        unsigned long long int*, // input
        unsigned long long int*, // local
        unsigned int             // etype
    );
};

#endif
