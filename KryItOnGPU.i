%module(directors="1") KryItOnGPU

%header
%{
#include <mpi.h>
%}


%feature("director") kry_ctx;

%inline %{

struct kry_ctx {
    virtual void UpdateRhs()=0;
    virtual void MatrixVecProduct()=0;
    virtual void Preconditioner()=0;
    virtual ~kry_ctx() {};
};

%}

%{
extern kry_ctx *kry_ptr;
%}

// initialize the struct

%{
kry_ctx* kry_ptr = NULL;
// core functions will be used to interact with PyFR
static void helper_UpdateRhs() {
    return kry_ptr->UpdateRhs();
}
static void helper_MatrixVecProduct() {
    return kry_ptr->MatrixVecProduct();
}
static void helper_Preconditioner() {
    return kry_ptr->Preconditioner();
};
%}


