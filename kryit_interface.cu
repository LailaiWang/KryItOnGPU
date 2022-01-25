#include "kryit_interface.h"
#include <stdio.h>
#include <unistd.h>
#include "Python.h"
/* \fn function to create the gmres context
 * @author Lai Wang
 */
void* create_gmres_ctx(unsigned int size,
                       unsigned int dim,
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
    
    int pid = getpid();
    printf("get current pid %d\n", pid);
    int idebugger = 0;
    while(idebugger) {

    }
    
    unsigned int* soasz = (unsigned int*) soasz_in;
    unsigned int* ioshape = (unsigned int*) ioshape_in;
    unsigned int* datashape = (unsigned int*) datashape_in;

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
            &set_b_vector_d
        );
        
        /*allocate the ram here*/
        gctx->allocate_ram(
            gctx->b,  gctx->Q,  gctx->h,  gctx->v, 
            gctx->sn, gctx->cs, gctx->e1, gctx->beta,
            gctx->xdim, 
            gctx->kspace
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
            &set_b_vector_f
        );
        /*allocate the ram here*/
        gctx->allocate_ram(
            gctx->b,  gctx->Q,  gctx->h,  gctx->v, 
            gctx->sn, gctx->cs, gctx->e1, gctx->beta,
            gctx->xdim, 
            gctx->kspace
        );

        return (void*) gctx;
    }
}

void set_gmres_b_vector(void* gmres, void* data, unsigned int size) {
    if(size == sizeof(double)) {
        struct gmres_app_ctx<double>* gctx = (struct gmres_app_ctx<double>*) (gmres);
        double *bvec = (double*) data;
        gctx->set_b_vector(bvec, gctx->b, gctx->xdim, gctx->bdim, gctx->bstrides);
    } else {
        struct gmres_app_ctx<float>* gctx  = (struct gmres_app_ctx<float>*) (gmres);
        float *bvec = (float*) data;
        gctx->set_b_vector(bvec, gctx->b, gctx->xdim,  gctx->bdim, gctx->bstrides);
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

void*  create_precon_ctx(unsigned int size, 
                         unsigned int xdim, 
                         void* xin,
                         void* resin) {
    if(size == sizeof(double)) {
        // cast pointer to 
        double* x   = (double*) (xin);
        double* res = (double*) (resin);
        struct precon_app_ctx<double>* pctx = new precon_app_ctx<double>(
            x, res, xdim
        );
        return (void *) pctx;
    } else if (size == sizeof(float)) {
        // cast pointer to 
        float* x   = (float*) (xin);
        float* res = (float*) (resin);
        struct precon_app_ctx<float>* pctx = new precon_app_ctx<float>(
            x, res, xdim
        );
        return (void*) pctx;
    }
}

void clean_precon_ctx(unsigned int size, void* input) {
    if(size == sizeof(double)) { 
        delete (struct precon_app_ctx<double>*) input;
        return;
    } else if (size == sizeof(float)) {
        delete (struct precon_app_ctx<float>*) input;
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
            void (*) (void*, void*, void*, unsigned int),
            void (*) (void*, void*, unsigned int),
            void*, 
            void*,
            void*,
            void*
        ) = &MFgmres<double>;
        
        return (void*) gmresSol;
    } else if(size == sizeof(float)) {
        void (*gmresSol) (
            void (*) (void*, void*, void*, unsigned int),
            void (*) (void*, void*, unsigned int),
            void*,
            void*,
            void*,
            void*
        ) = &MFgmres<float>;
        return (void* ) gmresSol;
    }
}

void gmres_solve(
                 void* matdotptr, // function pointers
                 void* preconptr, // function pointers
                 void* gmresptr,   // function pointers
                 void* solctx,    // solver context
                 void* pctxptr,   // structure pointers
                 void* gctxptr,   // structure pointers
                 void* bctxptr    // structure pointers
                ) {
    // cast the void pointer to the function pointer
    void (*funcdot) (
        void*,       // solctx
        void*,       // corresponding to B 
        void*,       // corresponding to X
        unsigned int // dimension of the problem
    ) = ( void(*) (
            void*,     
            void*,
            void*,
            unsigned int
          )
        ) matdotptr;
    
    void (*funcprecon) (
        void*,
        void*,
        unsigned int
    ) = ( void(*) (
           void*,       // solctx
           void*,       // vector to be preconditioner
           unsigned int // dimension of the problem
          )
        ) preconptr;


    void (*gmresSol) (
        void (*) (void*, void*, void*, unsigned int),
        void (*) (void*, void*, unsigned int),
        void*, // solver context
        void*, // preconditioner context
        void*, // gmres context
        void*  // cublas context
    ) = ( void (*) (
            void (*) (void*, void*, void*, unsigned int),
            void (*) (void*, void*, unsigned int),
            void*,
            void*,
            void*,
            void*)
        ) gmresptr;
    // solve the system using gmres   
    gmresSol(funcdot, funcprecon, solctx, pctxptr, gctxptr, bctxptr);
}


void set_zero(void* x, unsigned int xdim, unsigned int size) {
    if(size == sizeof(double)) {
        set_zeros_double((double*)x, xdim);
    } else {
        set_zeros_float ((float*)x, xdim);
    }
}

void set_one(void* x, unsigned int xdim, unsigned int size) {
    if(size == sizeof(double)) {
        set_ones_double((double*)x, xdim);
    } else {
        set_ones_float ((float*)x, xdim);
    }
}

void print_data(void* x, unsigned int xdim, unsigned int size) {
    if(size == sizeof(double)) {
        print_data_wrapper<double>((double*)x, xdim);
    } else {
        print_data_wrapper<float> ((float*)x, xdim);
    }

}

void (*matdot_d) ( void*, void*, unsigned int) = &MatDotVec_wrapper<double>;
void (*matdot_f) ( void*, void*, unsigned int) = &MatDotVec_wrapper<float>;

void  get_matdot(void* c, void* b, unsigned int xdim, unsigned int size) {
    if(size == sizeof(double)) {
        matdot_d(c,b,xdim);
    } else if (size == sizeof(float)) {
        matdot_f(c,b,xdim);
    }
}
