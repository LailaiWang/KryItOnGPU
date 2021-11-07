# KryItOnGPU
Krylov iterative methods on GPU
This is a project to create a library for matrix-free Krylov subpsace methods on GPU.
The preconditioner and matrix-vector product are supposed to be supplied by the user.
The library is callable from Python using swig.
We intend to develop this library for any CFD software runs on GPU to perform implicit time
integration using advanced preconditioners.
