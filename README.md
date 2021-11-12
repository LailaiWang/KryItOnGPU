# KryItOnGPU
Krylov iterative methods on GPU
This is a project to create a library for matrix-free Krylov subpsace methods on GPU.
The preconditioner and matrix-vector product are supposed to be supplied by the user.
The library is callable from Python using swig.
The software is developed for PyFR to perform implicit time integration or any other
CFD software runs on GPU.
