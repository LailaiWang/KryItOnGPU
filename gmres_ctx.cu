#include "gmres_ctx.cuh"
#include "cuda_constant.cuh"
#include <iostream>

void allocate_ram_gmres_app_ctx_d(
        double* &Q,  double* &h,  double* &v,
        double* &sn, double* &cs, double* &e1, double* &beta,
        bool* &nonpadding,
        unsigned long int xdim, unsigned int kspace
){
    
    cudaMalloc((void **) &Q,    sizeof(double)*xdim*(kspace+1));
    cudaMalloc((void **) &h,    sizeof(double)*kspace*(kspace+1));
    cudaMalloc((void **) &v,    sizeof(double)*xdim);

    cudaMalloc((void **) &sn,   sizeof(double)*(kspace+1));
    cudaMalloc((void **) &cs,   sizeof(double)*(kspace+1));
    cudaMalloc((void **) &e1,   sizeof(double)*(kspace+1));
    cudaMalloc((void **) &beta, sizeof(double)*(kspace+11));
    cudaMalloc((void **) &nonpadding, sizeof(bool)*xdim);
}
    
void deallocate_ram_gmres_app_ctx_d(
        double* &Q,  double* &h,  double* &v,
        double* &sn, double* &cs, double* &e1, double* &beta, 
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

void allocate_ram_gmres_app_ctx_f(
        float* &Q,  float* &h,  float* &v,
        float* &sn, float* &cs, float* &e1, float* &beta,
        bool* &nonpadding,
        unsigned long int xdim, unsigned int kspace
){
    

    cudaMalloc((void **) &Q,    sizeof(float )*xdim*(kspace+1));
    cudaMalloc((void **) &h,    sizeof(float )*kspace*(kspace+1));
    cudaMalloc((void **) &v,    sizeof(float )*xdim);

    cudaMalloc((void **) &sn,   sizeof(float )*(kspace+1));
    cudaMalloc((void **) &cs,   sizeof(float )*(kspace+1));
    cudaMalloc((void **) &e1,   sizeof(float )*(kspace+1));
    cudaMalloc((void **) &beta, sizeof(float )*(kspace+11));
    cudaMalloc((void **) &nonpadding, sizeof(bool)*xdim);
}
    
void deallocate_ram_gmres_app_ctx_f(
        float* &Q,  float* &h,  float* &v,
        float* &sn, float* &cs, float* &e1, float* &beta, bool* &nonpadding
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

void set_ones_const() {
    float  fno = -1.0;
    float  fpo =  1.0;
    double dno = -1.0;
    double dpo =  1.0;
    
    CUDA_CALL(cudaMemcpyToSymbol(P_ONE_F, &fpo, sizeof(float), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(N_ONE_F, &fno, sizeof(float), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(P_ONE_D, &dpo, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(N_ONE_D, &dno, sizeof(double), 0, cudaMemcpyHostToDevice));
    return;
}

void fill_nonpadding(unsigned int etype, 
                     unsigned long int* soasz,
                     unsigned int iodim,
                     unsigned long int* ioshape,
                     unsigned int datadim,
                     unsigned long int* datashape,
                     unsigned long int xdim,
                     bool* nonpadding){
    // sort out if an entry in the stage vector is padding or not
    
    // ioshape is   [nupts, nvar, neles]
    // datashape is [nblocks, nupts, nparr//(nblocks*soasz), nvar,soasz]
    bool* hnopad = (bool*) malloc(sizeof(bool)*xdim);
    
    unsigned long int isz = soasz[0];
    unsigned long int csz = soasz[1];

    if(datadim == 5) {
        for(unsigned int ie = 0; ie<etype;ie++) {
            unsigned long int d0 = datashape[ie*5+0];
            unsigned long int d1 = datashape[ie*5+1];
            unsigned long int d2 = datashape[ie*5+2];
            unsigned long int d3 = datashape[ie*5+3];
            unsigned long int d4 = datashape[ie*5+4];

            unsigned long int stride0 = d1*d2*d3*d4;
            unsigned long int stride1 = d2*d3*d4;
            unsigned long int stride2 = d3*d4;
            unsigned long int stride3 = d4;
            unsigned long int stride4 = 1;
            // first and third dimension could be pading
            for(unsigned long int i0=0;i0<d0;i0++) {
                for(unsigned long int i1=0;i1<d1;i1++) {
                    for(unsigned long int i2=0;i2<d2;i2++) {
                        for(unsigned long int i3=0;i3<d3;i3++) {
                            for(unsigned long int i4=0;i4<d4;i4++) {
                                
                                // first dimension is related to nelem
                                if(i0*d2*d4+i2*d4+i4 < ioshape[ie*3+2]) {
                                    hnopad[i0*stride0 +
                                           i1*stride1 +
                                           i2*stride2 +
                                           i3*stride3 +
                                           i4*stride4] = true;
                                } else {
                                    hnopad[i0*stride0 +
                                           i1*stride1 +
                                           i2*stride2 +
                                           i3*stride3 +
                                           i4*stride4] = false;

                                }

                                printf("is padding %d\n",
                                    hnopad[i0*stride0 +
                                           i1*stride1 +
                                           i2*stride2 +
                                           i3*stride3 +
                                           i4*stride4]
                                );
                            }
                        }
                    }
                }
            }
        }
    } else {
        printf("datadim != 5, terminate\n");
        exit(0);
    }
    
    // copy this to the device
    cudaMemcpy(nonpadding, hnopad, sizeof(bool)*xdim, cudaMemcpyHostToDevice);
    free(hnopad);
}

void copy_data_to_native_d(
          unsigned long long int* startingAddr, // input starting address
          unsigned long long int localAddr,  // local address
          unsigned int etype, // element type
          unsigned int datadim, // dimension of the datashape
          unsigned long int *datashapes // datashape
) {
    
    // Data on PyFR side for each element type are not necessarily contiguous
    // Data on native gmres side are contiguous
    double* local = (double*) (localAddr);

    unsigned long int esize[etype+1];
    esize[0] = 0;
    for(unsigned int ie=0;ie<etype;ie++) {
        esize[ie+1] = 1;
        for(unsigned int idm=0;idm<datadim;idm++) {
            esize[ie+1] *= datashapes[ie*datadim+idm];
        }
    }

    for(unsigned int ie=0;ie<etype;ie++) {
        double* estart = reinterpret_cast<double*> (startingAddr[ie]);
        unsigned long int offset = 0;
        for(unsigned int k=0;k<=ie;k++) {
            offset += esize[k];
        }

        CUDA_CALL(
            cudaMemcpy(
                local+offset, estart,
                sizeof(double)*esize[ie+1], cudaMemcpyDeviceToDevice
           )
        );
    }

}

void copy_data_to_native_f(
          unsigned long long int* startingAddr, // input starting address
          unsigned long long int localAddr,  // local address
          unsigned int etype, // element type
          unsigned int datadim,
          unsigned long int * datashapes // datashape
){




}

void copy_data_to_user_d(
          unsigned long long int* startingAddr, // input starting address
          unsigned long long int localAddr,  // local address
          unsigned int etype, // element type
          unsigned int datadim, 
          unsigned long int * datashapes// datashape
){

    // Data on PyFR side for each element type are not necessarily contiguous
    // Data on native gmres side are contiguous
    double* local = (double*) (localAddr);
    unsigned long int esize[etype+1];
    esize[0] = 0;
    for(unsigned int ie=0;ie<etype;ie++) {
        esize[ie+1] = 1;
        for(unsigned int idm=0;idm<datadim;idm++) {
            esize[ie+1] *= datashapes[ie*datadim+idm];
        }
    }
    

    for(unsigned int ie=0;ie<etype;ie++) {
        double* estart = reinterpret_cast<double*> (startingAddr[ie]);
        unsigned long int offset = 0;
        for(unsigned int k=0;k<=ie;k++) {
            offset += esize[k];
        }

        CUDA_CALL(
            cudaMemcpy(
                estart, local+offset,
                sizeof(double)*esize[ie+1], cudaMemcpyDeviceToDevice
            )
        );
    }
}

void copy_data_to_user_f(
          unsigned long long int* startingAddr, // input starting address
          unsigned long long int localAddr,  // local address
          unsigned int etype, // element type
          unsigned int datadim,
          unsigned long int *datashapes // datashape
){


}

/*setting the address of b vector on PyFR side*/
void set_reg_addr_pyfr(
    unsigned long long int* input, // input
    unsigned long long int* output, // local
    unsigned int etype// etype
) {
    for(unsigned int ie = 0; ie<etype;ie++) {
        output[ie] = input[ie];
        printf("current reg address is %ld\n", output[ie]);
    }
}


