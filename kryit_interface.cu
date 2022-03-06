#include "kryit_interface.h"
#include <stdio.h>
#include <unistd.h>
#include "Python.h"
/* \fn function to create the gmres context
 * @author Lai Wang
 */

void* create_gmres_ram_ctx(unsigned int dsize, unsigned long int xdim, unsigned int kspace) {
    if(dsize == sizeof(double)) {
        struct gmres_ram_ctx<double> *ram_ctx = new struct gmres_ram_ctx<double>(xdim, kspace);
        return (void*) ram_ctx;
    } else 
    if(dsize == sizeof(float)) {
        struct gmres_ram_ctx<float> *ram_ctx = new struct gmres_ram_ctx<float>(xdim, kspace);
        return (void*) ram_ctx;
    }
    return (void*) NULL;

}

void* create_expm_ram_ctx(unsigned int dsize, unsigned int ppade, unsigned int kspace) {
    if(dsize == sizeof(double)) {
        struct expm_ctx_ram<double>* expram_ctx = new struct expm_ctx_ram<double>(ppade, kspace);
        return (void*) expram_ctx;
    } else 
    if(dsize == sizeof(float)) {
        struct expm_ctx_ram<float>* expram_ctx = new struct expm_ctx_ram<float>(ppade, kspace);
        return (void*) expram_ctx;
    }
    return (void*) NULL;
}

void* create_expm_ctx(unsigned int dsize, 
                      unsigned int pin,
                      unsigned int din, 
                      void* ram_ctx_in,
                      void* stream_in) {
    if(dsize == sizeof(double)) {
        struct expm_ctx<double>* expm_app_ctx = new struct expm_ctx<double> (
                pin, din, ram_ctx_in, (cudaStream_t) stream_in
            );
        return (void*) expm_app_ctx;
    } else 
    if(dsize == sizeof(float) ) {
        struct expm_ctx<float>* expm_app_ctx = new struct expm_ctx<float> (
                pin, din, ram_ctx_in, (cudaStream_t) stream_in
            );
        return (void*) expm_app_ctx;
    }
    return (void*) NULL;
}

void* create_gmres_ctx(MPI_Comm mpicomm,
                       unsigned int nranks,
                       unsigned int size,
                       unsigned long int dim,
                       unsigned int etypes,   // number of element types
                       void* soasz_in,
                       unsigned int iodim,    // dim of ioshape
                       void* ioshape_in,         
                       unsigned int datadim, 
                       void* datashape_in,
                       unsigned int space,
                       void* atol,
                       void* rtol,
                       void* ram_ctx
                      ) {
    
    unsigned long int* soasz = (unsigned long int*) soasz_in;
    unsigned long int* ioshape = (unsigned long int*) ioshape_in;
    unsigned long int* datashape = (unsigned long int*) datashape_in;

    if (size == sizeof(double)) {
        double at = *((double*)(atol));
        double rt = *((double*)(rtol));
        struct gmres_ram_ctx<double>* ramctx = (struct gmres_ram_ctx<double>*) ram_ctx;
        struct gmres_app_ctx<double>* gctx = new gmres_app_ctx<double>(
            mpicomm, nranks,
            dim, etypes,
            soasz, 
            iodim, ioshape,
            datadim, datashape,
            space, at, rt, ramctx
        );
        return (void*) gctx;
    } else if(size == sizeof(float)) {
        float at = *((float*)(atol));
        float rt = *((float*)(rtol));
        struct gmres_ram_ctx<float>* ramctx = (struct gmres_ram_ctx<float>*) ram_ctx;
        struct gmres_app_ctx<float>* gctx = new gmres_app_ctx<float>(
            mpicomm, nranks,
            dim, etypes,
            soasz,
            iodim, ioshape,
            datadim, datashape,
            space, at, rt, ramctx
        );
        return (void*) gctx;
    }
    return (void*) NULL; /*something went wrong return NULL*/
}

void* create_gmres_expmctx(MPI_Comm mpicomm,
                           unsigned int nranks,
                           unsigned int size,
                           unsigned long int dim,
                           unsigned int etypes,   // number of element types
                           void* soasz_in,
                           unsigned int iodim,    // dim of ioshape
                           void* ioshape_in,         
                           unsigned int datadim, 
                           void* datashape_in,
                           unsigned int space,
                           void* atol,
                           void* rtol,
                           void* ram_ctx,
                           void* expmram_ctx_in,
                           void* expmctx_in
                          ) {
    
    unsigned long int* soasz = (unsigned long int*) soasz_in;
    unsigned long int* ioshape = (unsigned long int*) ioshape_in;
    unsigned long int* datashape = (unsigned long int*) datashape_in;

    if (size == sizeof(double)) {
        double at = *((double*)(atol));
        double rt = *((double*)(rtol));
        struct gmres_ram_ctx<double>* ramctx = (struct gmres_ram_ctx<double>*) ram_ctx;
        struct gmres_app_ctx<double>* gctx = new gmres_app_ctx<double>(
            mpicomm, nranks,
            dim, etypes,
            soasz, 
            iodim, ioshape,
            datadim, datashape,
            space, at, rt, ramctx, expmram_ctx_in, expmctx_in
        );
        return (void*) gctx;
    } else if(size == sizeof(float)) {
        float at = *((float*)(atol));
        float rt = *((float*)(rtol));
        struct gmres_ram_ctx<float>* ramctx = (struct gmres_ram_ctx<float>*) ram_ctx;
        struct gmres_app_ctx<float>* gctx = new gmres_app_ctx<float>(
            mpicomm, nranks,
            dim, etypes,
            soasz,
            iodim, ioshape,
            datadim, datashape,
            space, at, rt, ramctx, expmram_ctx_in, expmctx_in
        );
        return (void*) gctx;
    }
    return (void*) NULL; /*something went wrong return NULL*/
}


void clean_gmres_ctx(unsigned int size, void* input) {
    if(size == sizeof(double)) {
        struct gmres_app_ctx<double>* gmres_ctx = (struct gmres_app_ctx<double>*) input;
        gmres_ctx->~gmres_app_ctx();
        delete gmres_ctx;
        return;
    } else if (size == sizeof(float)) {
        struct gmres_app_ctx<float>* gmres_ctx = (struct gmres_app_ctx<float>*) input;
        gmres_ctx->~gmres_app_ctx();
        delete gmres_ctx;
        return;
    }
    return;
}

void* create_cublas_ctx( unsigned int size, void* stream_comp_0) {
    cudaStream_t comp_stream_0 = (cudaStream_t) stream_comp_0;
    struct cublas_app_ctx* bctx = new cublas_app_ctx(comp_stream_0);
    return (void*) bctx;
}

void clean_cublas_ctx(void* input) {
    // need to destroy cublas
    struct cublas_app_ctx* bctx = (struct cublas_app_ctx*) input;
    bctx->~cublas_app_ctx();
    delete bctx;
    return;
}

// instantiate the gmres instance
void* gmres(unsigned int size) {
    
    if(size == sizeof(double)) {
        void (*gmresSol) (
            void (*) (void*, bool), /*solv_ctx, bool*/
            void*,
            void*,
            void*,
            unsigned int
        ) = &MFgmres<double>;
        
        return (void*) gmresSol;
    } else if(size == sizeof(float)) {
        void (*gmresSol) (
            void (*) (void*, bool), /*solv_ctx, bool*/
            void*,
            void*,
            void*,
            unsigned int
        ) = &MFgmres<float>;
        return (void* ) gmresSol;
    }

    return (void*) NULL; /*something went wrong return NULL*/
}


void* expstep(unsigned int size) {
    
    if(size == sizeof(double)) {
        void (*expSol) (
            void (*) (void*, bool), /*solv_ctx, bool*/
            void*,
            void*,
            void*,
            unsigned int
        ) = &MFexponential<double>;
        
        return (void*) expSol;
    } else if(size == sizeof(float)) {
        void (*expSol) (
            void (*) (void*, bool), /*solv_ctx, bool*/
            void*,
            void*,
            void*,
            unsigned int
        ) = &MFexponential<float>;
        return (void* ) expSol;
    }

    return (void*) NULL; /*something went wrong return NULL*/
}

void gmres_solve(
                 void* matdotptr, // function pointers
                 void* gmresptr,   // function pointers
                 void* solctx,    // solver context
                 void* gctxptr,   // structure pointers
                 void* bctxptr,    // structure pointers
                 unsigned int icnt
                ) {
    // cast the void pointer to the function pointer
    void (*funcdot) (
        void*,       // solv_ctx
        bool         // bool
    ) = ( void(*) (
            void*,     
            bool
          )
        ) matdotptr;
    
    void (*gmresSol) (
        void (*) (void*, bool), /*solv ctx, bool*/
        void*, // solver context
        void*, // gmres context
        void*, // cublas context
        unsigned int
    ) = ( void (*) (
            void (*) (void*, bool),
            void*,
            void*,
            void*,
            unsigned int)
        ) gmresptr;
    // solve the system using gmres   
    gmresSol(funcdot, solctx, gctxptr, bctxptr, icnt);
}

void exp_solve(
                 void* matdotptr, // function pointers
                 void* expptr,   // function pointers
                 void* solctx,    // solver context
                 void* gctxptr,   // structure pointers
                 void* bctxptr,    // structure pointers
                 unsigned int icnt
                ) {
    // cast the void pointer to the function pointer
    void (*funcdot) (
        void*,       // solv_ctx
        bool         // bool
    ) = ( void(*) (
            void*,     
            bool
          )
        ) matdotptr;
    
    void (*expSol) (
        void (*) (void*, bool), /*solv ctx, bool*/
        void*, // solver context
        void*, // gmres context
        void*, // cublas context
        unsigned int
    ) = ( void (*) (
            void (*) (void*, bool),
            void*,
            void*,
            void*,
            unsigned int)
        ) expptr;
    // solve the system using gmres   
    expSol(funcdot, solctx, gctxptr, bctxptr, icnt);
}


void print_data(void* x, unsigned long int xdim, unsigned int size) {
    if(size == sizeof(double)) {
        print_data_wrapper<double>((double*)x, xdim);
    } else {
        print_data_wrapper<float> ((float*)x, xdim);
    }

}

/*set the address for b vector*/
void set_b_reg_addr(void* gmres,
                    void* addr,
                    unsigned int etype,
                    unsigned int dsize){
    unsigned long long int* data = (unsigned long long int *) addr;
    if(dsize == sizeof(double)) {
        struct gmres_app_ctx<double>* gctx = (struct gmres_app_ctx<double>*) (gmres);
        gctx->set_reg_addr(data, gctx->b_reg, etype);
    } else {
        struct gmres_app_ctx<float>* gctx  = (struct gmres_app_ctx<float>*) (gmres);
        gctx->set_reg_addr(data, gctx->b_reg, etype);
    }
}

/*set the address for x vector*/
void set_x_reg_addr(void* gmres,
                    void* addr,
                    unsigned int etype,
                    unsigned int dsize){
    unsigned long long int* data = (unsigned long long int *) addr;
    if(dsize == sizeof(double)) {
        struct gmres_app_ctx<double>* gctx = (struct gmres_app_ctx<double>*) (gmres);
        gctx->set_reg_addr(data, gctx->x_reg, etype);
    } else {
        struct gmres_app_ctx<float>* gctx  = (struct gmres_app_ctx<float>*) (gmres);
        gctx->set_reg_addr(data, gctx->x_reg, etype);
    }
}

/*set the address for current operating space*/
void set_curr_reg_addr(void* gmres,
                       void* addr,
                       unsigned int etype,
                       unsigned int dsize){
    unsigned long long int* data = (unsigned long long int *) addr;
    if(dsize == sizeof(double)) {
        struct gmres_app_ctx<double>* gctx = (struct gmres_app_ctx<double>*) (gmres);
        gctx->set_reg_addr(data, gctx->curr_reg, etype);
    } else {
        struct gmres_app_ctx<float>* gctx  = (struct gmres_app_ctx<float>*) (gmres);
        gctx->set_reg_addr(data, gctx->curr_reg, etype);
    }
}

void check_b_reg_data(void* gmres, unsigned int ne, unsigned int len, unsigned int dsize) {
    if(dsize == sizeof(double)) {
        struct gmres_app_ctx<double>* gctx = (struct gmres_app_ctx<double>*) (gmres);
        print_data_wrapper<double> (reinterpret_cast<double*>(gctx->b_reg[ne]), len);   
    } else {
        struct gmres_app_ctx<float>* gctx  = (struct gmres_app_ctx<float>*) (gmres);
        print_data_wrapper<float> (reinterpret_cast<float*>(gctx->b_reg[ne]), len); 
    }
}

void check_x_reg_data(void* gmres, unsigned int ne, unsigned int len, unsigned int dsize) {
    if(dsize == sizeof(double)) {
        struct gmres_app_ctx<double>* gctx = (struct gmres_app_ctx<double>*) (gmres);
        print_data_wrapper<double> (reinterpret_cast<double*>(gctx->x_reg[ne]), len);   
    } else {
        struct gmres_app_ctx<float>* gctx  = (struct gmres_app_ctx<float>*) (gmres);
        print_data_wrapper<float> (reinterpret_cast<float*>(gctx->x_reg[ne]), len); 
    }
}

