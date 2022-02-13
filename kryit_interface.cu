#include "kryit_interface.h"
#include <stdio.h>
#include <unistd.h>
#include "Python.h"
/* \fn function to create the gmres context
 * @author Lai Wang
 */
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
                       void* rtol
                      ) {
    
    unsigned long int* soasz = (unsigned long int*) soasz_in;
    unsigned long int* ioshape = (unsigned long int*) ioshape_in;
    unsigned long int* datashape = (unsigned long int*) datashape_in;

    if (size == sizeof(double)) {
        double at = *((double*)atol);
        double rt = *((double*)rtol);
        
        // testing mpi_common_world
        int size, rank;
        char pname[MPI_MAX_PROCESSOR_NAME]; int len;
        if (mpicomm == MPI_COMM_NULL) {
            printf("You passed MPI_COMM_NULL !!!\n");
        }
        MPI_Comm_size(mpicomm, &size);
        MPI_Comm_rank(mpicomm, &rank);
        MPI_Get_processor_name(pname, &len);
        pname[len] = 0;
        printf("Hello, World! I am process %d of %d on %s.\n", rank, size, pname);

        struct gmres_app_ctx<double>* gctx = new gmres_app_ctx<double>(
            mpicomm, nranks,
            dim, etypes,
            soasz, 
            iodim, ioshape,
            datadim, datashape,
            space, at, rt,
            &allocate_ram_gmres_app_ctx<double>,
            &deallocate_ram_gmres_app_ctx<double>,
            &fill_nonpadding,
            &copy_data_to_native<double>,
            &copy_data_to_user<double>,
            &set_reg_addr_pyfr
        );
        
        /*allocate the ram here*/
        gctx->allocate_ram(
            gctx->Q,  gctx->h,  gctx->v, 
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
            mpicomm, nranks,
            dim, etypes,
            soasz,
            iodim, ioshape,
            datadim, datashape,
            space, at, rt,
            &allocate_ram_gmres_app_ctx<float>,
            &deallocate_ram_gmres_app_ctx<float>,
            &fill_nonpadding,
            &copy_data_to_native<float>,
            &copy_data_to_user<float>,
            &set_reg_addr_pyfr
        );
        /*allocate the ram here*/
        gctx->allocate_ram(
            gctx->Q,  gctx->h,  gctx->v, 
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
    return (void*) NULL; /*something went wrong return NULL*/
}


void clean_gmres_ctx(unsigned int size, void* input) {
    if(size == sizeof(double)) {
        delete (struct gmres_app_ctx<double>*) input;
        return;
    } else if (size == sizeof(float)) {
        delete (struct gmres_app_ctx<float>*) input;
        return;
    }
    return;
}

void* create_cublas_ctx(
    unsigned int size,
    void* stream_comp_0,
    void* stream_copy_0,
    void* stream_comp_1,
    void* stream_copy_1) {
    // cast the pointer
    
    cudaStream_t comp_stream_0 = (cudaStream_t) stream_comp_0;
    cudaStream_t copy_stream_0 = (cudaStream_t) stream_copy_0;
    cudaStream_t comp_stream_1 = (cudaStream_t) stream_comp_1;
    cudaStream_t copy_stream_1 = (cudaStream_t) stream_copy_1;
    
    // need to launch cublas
    struct cublas_app_ctx* bctx = 
        new cublas_app_ctx(
            comp_stream_0,
            copy_stream_0,
            comp_stream_1,
            copy_stream_1,
            &initialize_cublas,
            &finalize_cublas);
    bctx->create_cublas(bctx->handle);
    // for each handle assign a stream to it
    cublasSetStream(bctx->handle[0], bctx->comp_stream_0);
    cublasSetStream(bctx->handle[1], bctx->copy_stream_0);
    cublasSetStream(bctx->handle[2], bctx->comp_stream_1);
    cublasSetStream(bctx->handle[3], bctx->copy_stream_1);

    return (void*) bctx;
}

void clean_cublas_ctx(void* input) {
    // need to destroy cublas
    struct cublas_app_ctx* bctx = (struct cublas_app_ctx*) input;
    bctx->clean_cublas(bctx->handle);
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

    return (void*) NULL; /*something went wrong return NULL*/
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

void check_curr_reg_data(void* gmres, unsigned int ne, unsigned int len, unsigned int dsize) {
    if(dsize == sizeof(double)) {
        struct gmres_app_ctx<double>* gctx = (struct gmres_app_ctx<double>*) (gmres);
        print_data_wrapper<double> (reinterpret_cast<double*>(gctx->curr_reg[ne]), len);   
    } else {
        struct gmres_app_ctx<float>* gctx  = (struct gmres_app_ctx<float>*) (gmres);
        print_data_wrapper<float> (reinterpret_cast<float*>(gctx->curr_reg[ne]), len); 
    }
}

void dot_c_reg(void* gmres, void* cublas, unsigned int dsize) {
    struct cublas_app_ctx* bctx = (struct cublas_app_ctx*) (cublas);
    if(dsize == sizeof(double)) {
        double dotproduct = 0.0;
        mGPU_dot_creg_wrapper<double>(gmres, &dotproduct, bctx->handle);
        printf("inner product is %lf\n", dotproduct);
    } else {
        float dotproduct = 0.0;
        mGPU_dot_creg_wrapper<float>(gmres, &dotproduct, bctx->handle);
        printf("inner product is %f\n", dotproduct);
    }
}

void dot_b_reg(void* gmres, void* cublas, unsigned int dsize) {
    struct cublas_app_ctx* bctx = (struct cublas_app_ctx*) (cublas);
    if(dsize == sizeof(double)) {
        double dotproduct = 0.0;
        mGPU_dot_breg_wrapper<double>(gmres, &dotproduct, bctx->handle);
        printf("inner product is %lf\n", dotproduct);
    } else {
        float dotproduct = 0.0;
        mGPU_dot_breg_wrapper<float>(gmres, &dotproduct, bctx->handle);
        printf("inner product is %f\n", dotproduct);
    }
}
