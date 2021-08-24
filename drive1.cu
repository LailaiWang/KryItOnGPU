#include "arnoldi.cuh"
#include "MatDotVec.cuh"
#include "stdio.h"


// instantiate the function for debugging purpose
void (*print_on_gpu) (double*, unsigned int) = &print_data_wrapper<double>;

int driver1() {
    
    unsigned int n = 6;
    unsigned int m = 5;
    double *b = (double *) malloc(n*sizeof(double));
    double *q = (double *) malloc(n*sizeof(double));
    double *Q = (double *) malloc(n*m*sizeof(double));
    double *h = (double *) malloc(m*(m+1)*sizeof(double));
    double *v = (double *) malloc(n*sizeof(double));

    double *b_d;
    double *q_d;
    double *Q_d;
    double *h_d;
    double *v_d;
    
    cudaMalloc((void**) &b_d, n*sizeof(double));
    cudaMalloc((void**) &q_d, n*sizeof(double));
    cudaMalloc((void**) &Q_d, n*m*sizeof(double));
    cudaMalloc((void**) &h_d, m*(m+1)*sizeof(double));
    cudaMalloc((void**) &v_d, n*sizeof(double));

    for (int i=0;i<n;i++) {
        b[i] = 1.0;
    }

    cudaMemcpy(b_d, b, n*sizeof(double), cudaMemcpyHostToDevice);
    
    // initialize the function pointer
    void (*fn) (double*, double*, unsigned int) = &MatDotVec_wrapper<double>;
    arnoldi<double>(fn, b_d, q_d, Q_d, h_d, v_d, n, m);
    
    cudaMemcpy(Q, Q_d, n*m*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h, h_d, m*(m+1)*sizeof(double), cudaMemcpyDeviceToHost);

    free(b);
    free(q);
    free(Q);
    free(h);
    free(v);

    cudaFree(b_d);
    cudaFree(q_d);
    cudaFree(Q_d);
    cudaFree(h_d);
    cudaFree(v_d);
    
    return 0;
}

