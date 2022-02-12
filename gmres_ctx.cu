#include "gmres_ctx.cuh"
#include "cuda_constant.cuh"
#include <iostream>


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
    
    //unsigned long int isz = soasz[0];
    //unsigned long int csz = soasz[1];

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


/*setting the address of b vector on PyFR side*/
void set_reg_addr_pyfr(
    unsigned long long int* input, // input
    unsigned long long int* output, // local
    unsigned int etype// etype
) {
    for(unsigned int ie = 0; ie<etype;ie++) {
        output[ie] = input[ie];
    }
}


