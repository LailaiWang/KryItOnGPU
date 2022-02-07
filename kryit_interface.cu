#include "kryit_interface.h"
#include <stdio.h>
#include <unistd.h>
#include "Python.h"
/* \fn function to create the gmres context
 * @author Lai Wang
 */
void* create_gmres_ctx(unsigned int size,
                       unsigned long int dim,
                       unsigned int etypes,   // number of element types
                       void* soasz_in,
                       unsigned int iodim,    // dim of ioshape
                       void* ioshape_in,         
                       unsigned int datadim, 
                       void* datashape_in,
                       unsigned int space,
                       void* atol,
                       void* rtol
                      ) {
    
    unsigned long int* soasz = (unsigned long int*) soasz_in;
    unsigned long int* ioshape = (unsigned long int*) ioshape_in;
    unsigned long int* datashape = (unsigned long int*) datashape_in;

    if (size == sizeof(double)) {
        double at = *((double*)atol);
        double rt = *((double*)rtol);
        
        struct gmres_app_ctx<double>* gctx = new gmres_app_ctx<double>(
            dim, etypes,
            soasz, 
            iodim, ioshape,
            datadim, datashape,
            space, at, rt,
            &allocate_ram_gmres_app_ctx_d,
            &deallocate_ram_gmres_app_ctx_d,
            &fill_nonpadding,
            &copy_data_to_native_d,
            &copy_data_to_user_d,
            &set_reg_addr_pyfr
        );
        
        /*allocate the ram here*/
        gctx->allocate_ram(
            gctx->b,  gctx->Q,  gctx->h,  gctx->v, 
            gctx->sn, gctx->cs, gctx->e1, gctx->beta, gctx->nonpadding,
            gctx->xdim, 
            gctx->kspace
        );
        
        gctx->fill(
            gctx->etypes, gctx->soasz, gctx->iodim, gctx->ioshape,
            gctx->datadim, gctx->datashape, gctx->xdim, gctx->nonpadding
        );

        return (void*) gctx;
    } else if(size == sizeof(float)) {
        float at = *((float*)(atol));
        float rt = *((float*)(rtol));
        struct gmres_app_ctx<float>* gctx = new gmres_app_ctx<float>(
            dim, etypes,
            soasz,
            iodim, ioshape,
            datadim, datashape,
            space, at, rt,
            &allocate_ram_gmres_app_ctx_f,
            &deallocate_ram_gmres_app_ctx_f,
            &fill_nonpadding,
            &copy_data_to_native_f,
            &copy_data_to_user_f,
            &set_reg_addr_pyfr
        );
        /*allocate the ram here*/
        gctx->allocate_ram(
            gctx->b,  gctx->Q,  gctx->h,  gctx->v, 
            gctx->sn, gctx->cs, gctx->e1, gctx->beta, gctx->nonpadding,
            gctx->xdim, 
            gctx->kspace
        );

        gctx->fill(
            gctx->etypes, gctx->soasz, gctx->iodim, gctx->ioshape,
            gctx->datadim, gctx->datashape,  gctx->xdim, gctx->nonpadding
        );
        return (void*) gctx;
    }
}


void clean_gmres_ctx(unsigned int size, void* input) {
    if(size == sizeof(double)) {
        delete (struct gmres_app_ctx<double>*) input;
        return;
    } else if (size == sizeof(float)) {
        delete (struct gmres_app_ctx<float>*) input;
        return;
    }
}

void* create_cublas_ctx(unsigned int size) {
    // need to launch cublas
    struct cublas_app_ctx* bctx = new cublas_app_ctx(&initialize_cublas, &finalize_cublas);
    bctx->create_cublas(&bctx->handle);
    return (void*) bctx;
}

void clean_cublas_ctx(void* input) {
    // need to destroy cublas
    struct cublas_app_ctx* bctx = (struct cublas_app_ctx*) input;
    bctx->clean_cublas(&bctx->handle);
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
            void*
        ) = &MFgmres<double>;
        
        return (void*) gmresSol;
    } else if(size == sizeof(float)) {
        void (*gmresSol) (
            void (*) (void*, bool), /*solv_ctx, bool*/
            void*,
            void*,
            void*
        ) = &MFgmres<float>;
        return (void* ) gmresSol;
    }
}

void gmres_solve(
                 void* matdotptr, // function pointers
                 void* gmresptr,   // function pointers
                 void* solctx,    // solver context
                 void* gctxptr,   // structure pointers
                 void* bctxptr    // structure pointers
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
        void*  // cublas context
    ) = ( void (*) (
            void (*) (void*, bool),
            void*,
            void*,
            void*)
        ) gmresptr;
    // solve the system using gmres   
    gmresSol(funcdot, solctx, gctxptr, bctxptr);
}


void set_zero(void* x, unsigned long int xdim, unsigned int size) {
    if(size == sizeof(double)) {
        set_zeros_double((double*)x, xdim);
    } else {
        set_zeros_float ((float*)x, xdim);
    }
}

void set_one(void* x, unsigned long int xdim, unsigned int size) {
    if(size == sizeof(double)) {
        set_ones_double((double*)x, xdim);
    } else {
        set_ones_float ((float*)x, xdim);
    }
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
        gctx->set_reg_addr(data, gmres->b_reg, etype);
    } else {
        struct gmres_app_ctx<float>* gctx  = (struct gmres_app_ctx<float>*) (gmres);
        gctx->set_reg_addr(data, gmres->b_reg, etype);
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
        gctx->set_reg_addr(data, gmres->curr_reg, etype);
    } else {
        struct gmres_app_ctx<float>* gctx  = (struct gmres_app_ctx<float>*) (gmres);
        gctx->set_reg_addr(data, gmres->curr_reg, etype);
    }
}
