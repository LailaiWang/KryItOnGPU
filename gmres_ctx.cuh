#ifndef GMRES_CTX_CUH
#define GMRES_CTX_CUH

// passing pointer around
template<typename T>
struct {
    T* b;
    T* q;
    T* Q;
    T* h;
    T* v;

    unsigned int xdim = 0;    // dimension of the square matrix
    unsigned int kspace = 0;  // dimension of the krylov subpspace

    T atol;
    T rtol;

} gmres_ctx;

#endif
