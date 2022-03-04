#ifndef GMRES_CTX_CUH
#define GMRES_CTX_CUH

#include "cuda_runtime.h"
#include "util.cuh"

#include "mpi.h"
#include "mpi-ext.h"
#if defined(MPIX_CUDA_AWARE_SUPPORT)
#include "mpi-ext.h"
#endif

void set_ones_const();
enum gmres_conv_reason {GMRES_NOT_CONV = 0,
                        GMRES_CONV_ABS = 1,
                        GMRES_CONV_REL = 2,
                        GMRES_CONV_MAX_ITER = 3,
                        GMRES_DIV=-1};

template<typename T>
void allocate_ram_gmres_app_ctx(
        T* &Q,    T* &h,   T* &v,
        T* &sn,   T* &cs,  T* &beta,
        unsigned long int xdim, unsigned int kspace
){
    
    CUDA_CALL(cudaMalloc((void **) &Q,    sizeof(T)*xdim*(kspace+1));)
    CUDA_CALL(cudaMalloc((void **) &v,    sizeof(T)*xdim);)
    CUDA_CALL(cudaMalloc((void **) &h,    sizeof(T)*kspace*(kspace+1));)

    CUDA_CALL(cudaMalloc((void **) &sn,   sizeof(T)*(kspace+1));)
    CUDA_CALL(cudaMalloc((void **) &cs,   sizeof(T)*(kspace+1));)
    CUDA_CALL(cudaMalloc((void **) &beta, sizeof(T)*(kspace+11));)

    printf("done ram allocate\n");
}

template<typename T>
void deallocate_ram_gmres_app_ctx(
        T* &Q,   T* &h,   T* &v,
        T* &sn,  T* &cs,  T* &beta
){
    cudaFree(Q);
    cudaFree(h);
    cudaFree(v);

    cudaFree(sn);
    cudaFree(cs);
    cudaFree(beta);
}


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


/*move the ram outside such that it can be reused for gmres at different levels*/
template<typename T>
struct gmres_ram_ctx {
    unsigned long int xdim = 0;    // dimension of the square matrix
    unsigned int kspace = 0;  // dimension of the krylov subpspace max iteration number
    
    /*only need basic info for this*/

    /*all of these case be used at different levels*/
    T* Q;
    T* h;
    T* v;

    T* sn;
    T* cs;
    T* beta; /*last 10 for temp operation on GPU*/

    /*function pointer to deal with the ram*/
    void (*allocate_ram) ( T* &, T* &, T* &, T* &, T* &, T* &, unsigned long int, unsigned int);
    void (*deallocate_ram) ( T* &, T* &, T* &, T* &, T* &, T* &);
    
    /*a small constructor*/
    gmres_ram_ctx( unsigned long int dim, unsigned int space ) {
        xdim   = dim; 
        kspace = space;
        /*work out some function pointers*/
        allocate_ram = &allocate_ram_gmres_app_ctx<T>;
        deallocate_ram = &deallocate_ram_gmres_app_ctx<T>;

        /*allocate the ram*/
        allocate_ram_gmres_app_ctx(Q,h,v,sn,cs,beta, xdim, kspace);
    };
    
    /*a small destructor*/
    ~gmres_ram_ctx() {
        deallocate_ram(Q,h,v,sn,cs,beta);
        allocate_ram = nullptr;
        deallocate_ram = nullptr;
    };

};

// passing pointer around
template<typename T> 
struct gmres_app_ctx {
    MPI_Comm mpicomm;    // information of MPI
    unsigned int nranks;

    unsigned long int xdim = 0;    // dimension of the square matrix
    unsigned int kspace = 0;       // dimension of the krylov subpspace max iteration number

    unsigned int etypes;              // maximum tri quad hex tet prism pyrimid
    unsigned long int soasz[2];
    unsigned int iodim;
    unsigned long int ioshape[200];   // [nsp, nvar, neles]
    unsigned int datadim;
    unsigned long int datashape[200]; // [nblocks, nsp, neles/nblocks/soasz, nvar, soasz]
    
    unsigned int nconv = 0; // number of iteration upon convergence
    T atol=1e-11;
    T rtol=1e-10;
    
    gmres_conv_reason convrson = GMRES_NOT_CONV;
    unsigned int maxiters = 3500;
    unsigned int conv_iters = 0;

    unsigned long long int b_reg[10];    // starting address for b vector on PyFR side
    unsigned long long int x_reg[10];    // x soln on PyFR side
    unsigned long long int curr_reg[10]; // starting address for current operating space on PyFR
    
    /*keep these as local*/
    bool* nonpadding; // add a padding value for gmres calculation, skip padding space 
    T* abys; /*store some constants on Device for cublas*/
    
    /*these are just pointers which will pointer to the actual ram in ram_ctx*/
    T* Q;
    T* h;
    T* v;

    T* sn;
    T* cs;
    T* beta; /*last 10 for temp operation on GPU*/
    
    struct gmres_ram_ctx<T>* ram_ctx;

    void (*fill) (unsigned int, unsigned long int*, 
                  unsigned int, unsigned long int*,
                  unsigned int, unsigned long int*,
                  unsigned long int, bool*);

    void (*copy_to_native)(
          unsigned long long int*, // input starting address
          unsigned long long int,  // local address
          unsigned int,            // element type
          unsigned int,            // datadim
          unsigned long int *,     // datashape
          cudaStream_t
    );
    void (*copy_to_user) (
          unsigned long long int*, // input starting address
          unsigned long long int,  // local address
          unsigned int,            // element type
          unsigned int,            // datadim
          unsigned long int *,     // datashape
          cudaStream_t
    );
    void (*set_reg_addr) (
        unsigned long long int*, // input
        unsigned long long int*, // local
        unsigned int             // etype
    );
    
    std::vector<T> (* _view_content) (T* input, unsigned int cnts);

    /*change the constructor to a more concise one*/
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
                  void* ram_ctx_in
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
        
        /*assign the pointers to actual ram*/
        ram_ctx = (struct gmres_ram_ctx<T>*) ram_ctx_in;
        Q = ram_ctx->Q;
        h = ram_ctx->h;
        v = ram_ctx->v;

        sn = ram_ctx->sn;
        cs = ram_ctx->cs;
        beta = ram_ctx->beta;
        
        copy_to_native = &copy_data_to_native<T>;
        copy_to_user   = &copy_data_to_user<T>;
        
        _view_content = &view_content<T>;

        fill           = &fill_nonpadding;
        set_reg_addr   = &set_reg_addr_pyfr;

        /*push some coefficients to GPU*/
        T alps []= {(T) 1.0, (T) 0.0, (T) -1.0};
        cudaMalloc((void**)&abys, sizeof(T)*64);
        cudaMemcpy(abys, alps, sizeof(T)*3, cudaMemcpyHostToDevice);
        
        /*bool for if a entry in the array is for padding or not*/
        fill(etypes, soasz, iodim, ioshape, datadim, datashape, xdim, nonpadding);
    };
    
    /*a small destrutor*/
    ~gmres_app_ctx() {
        /*clean the ram*/
        cudaFree(abys);
        cudaFree(nonpadding);

        copy_to_native = nullptr;
        copy_to_user   = nullptr;

        set_reg_addr   = nullptr;
        fill           = nullptr;
        Q  = nullptr;
        h  = nullptr;
        v  = nullptr;
        sn = nullptr;
        cs = nullptr;
        beta = nullptr;
    }

};

#endif
