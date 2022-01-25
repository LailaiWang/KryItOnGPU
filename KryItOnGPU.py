from ctypes import c_void_p, c_uint, py_object, CFUNCTYPE, cdll, c_float, c_double, cast, byref, POINTER
import os
import numpy as np

cudart = cdll.LoadLibrary("libcudart.so")
cublas = cdll.LoadLibrary("libcublas.so")
kryit  = cdll.LoadLibrary("libkryitongpu.so")

@CFUNCTYPE(None, py_object, c_void_p, c_void_p, c_uint)
def MFmatvecapp(solv, C, B, n ):
    '''
    This is the subroutine to use the solver context
    to do the matrix-vector product A*X=B is performed here
    solv: the solver context, 
    B: the outut vector
    X: input vector
    Algorithm: Ab, A^2b, A^3b,.......A^(kdim-1)b 
    '''
    solv.perform_matdot(C, B)

@CFUNCTYPE(None, py_object, c_void_p, c_uint)
def PMGPrecond(solv, Y, n):
    '''
    This is the subroutine to use the solver context 
    to do the preconditioning
    solv: the solver context
    Y: the vector to be preconditioned
    -----------------------------------------------------------
    Algorithm: see paper nonlinear p-multigrid preconditioner 
    for implicit time integration of the Navier--Stokes equations
    -----------------------------------------------------------
    '''
    pass

class TestSolver(object):

    def __init__(self, xdim):
        '''
        initialize our solver
        '''
        self.xdim  = xdim
        self.dsize = 8 
        
        # initialize some functions
        self._ram_util()
        self._one()
        self._zero()
        self._print()

        self._create_var()
        self._matdot()
    
    def _one(self):
        self.set_ones = kryit.set_one
        self.set_ones.argtypes = [c_void_p, c_uint, c_uint]
        self.restype = None
    
    def _zero(self):
        self.set_zeros = kryit.set_zero
        self.set_zeros.argtypes = [c_void_p, c_uint, c_uint]
        self.restype = None

    def _print(self):
        self.print_data = kryit.print_data
        self.print_data.argtypes = [c_void_p, c_uint, c_uint]
        self.restype = None

    def _ram_util(self):
        
        self.create = cudart.cudaMalloc
        self.create.argtypes = [POINTER(c_void_p), c_uint]

    def _create_var(self):
        
        self.x   = c_void_p()
        self.res = c_void_p()
        
        # create the ram for x and res
        # x stores the solution
        # res is a vector to store b - Ax
        self.create(byref(self.x),   c_uint(self.dsize*self.xdim))
        self.create(byref(self.res), c_uint(self.dsize*self.xdim))
        
        # initialize # use 0 as the initial guess
        self.set_ones(self.x, c_uint(self.xdim), c_uint(self.dsize))
        self.set_zeros(self.res, c_uint(self.xdim), c_uint(self.dsize))
        
    def _matdot(self):
        '''
        Interface to do matrix vecotr product 
        Use current one as a simple example
        '''
        matdot = kryit.get_matdot
        matdot.argtypes = [c_void_p, c_void_p, c_uint, c_uint]
        matdot.restype  = None
        self.matdot = matdot

    def perform_matdot(self, output_vec, input_vec):
        '''
        function to perform the mat vec dot approximation
        first  input: output vector
        second input: input vector
        '''
        self.matdot(output_vec, input_vec, c_uint(self.xdim), c_uint(self.dsize))

class KryItOnGPU(object):

    def __init__(self, solctx):
        # store the solver app context
        self.solctx = solctx

        if solctx.dsize == 8 :
            self.dsize = c_uint(8)
        else:
            self.dsize = c_uint(4)

        self.xdim  = c_uint(solctx.xdim)
        self.space = c_uint(100)

        if solctx.dsize == 8:
            self.atol  = c_double(1e-5)
            self.rtol  = c_double(1e-5)
        else:
            self.atol  = c_float(1e-5)
            self.rtol  = c_float(1e-5)
        
        
        # create the gmres app context
        self._create_gmres_ctx()
        # create the preconditioner app context
        self._create_precon_ctx()
        # create the cublas app context
        self._create_cublas_ctx()
        # get gmres
        self._get_gmres()
        # get the solver
        self._get_solver()
        # self._set_b_vector
        self._set_b_vector()

        self.set_gmres_b(self.solctx.x)
        self.solctx.set_zeros(self.solctx.x, c_uint(self.solctx.xdim), c_uint(self.solctx.dsize))
        
    def _create_gmres_ctx(self):
        '''
        Interface to create the gmres application context
        Note that in gmres ctx, variable b stores the b in Ax=b
        '''
        gmres_ctx = kryit.create_gmres_ctx

        gmres_ctx.argtypes = [ c_uint, c_uint, c_uint, c_void_p, c_void_p]
        gmres_ctx.restype  = c_void_p

        self.gmres_ctx = gmres_ctx
        
        # return value is the address of the structure of gmres app context
        self.gctx = self.gmres_ctx(
            self.dsize, self.xdim, self.space,
            cast(byref(self.atol),c_void_p), cast(byref(self.rtol),c_void_p)
        )

    def _set_b_vector(self):
        setb = kryit.set_gmres_b_vector
        setb.argtypes = [c_void_p, c_void_p, c_uint]
        setb.restype = None

        self.set_b = setb
    
    def set_gmres_b(self, bvec):
        
        self.set_b(self.gctx, bvec, self.dsize)

    def _create_precon_ctx(self):
        '''
        Interface to create preconditioning application context
        '''
        precon_ctx = kryit.create_precon_ctx
        precon_ctx.argtypes = [c_uint, c_uint, c_void_p, c_void_p]
        precon_ctx.restype  = c_void_p

        self.precon_ctx = precon_ctx

        # return value is the address of the structure of the preconditioner
        # app context
        self.pctx = self.precon_ctx(
            self.dsize, self.xdim, 
            self.solctx.x,
            self.solctx.res
        )

    def _create_cublas_ctx(self):
        '''
        interface to create cublas application context
        '''
        cublas_ctx = kryit.create_cublas_ctx
        cublas_ctx.argtypes = [c_uint]
        cublas_ctx.restype = c_void_p
        self.cublas_ctx = cublas_ctx
        
        # return vaue is the address of the structure of the cublas app context
        self.bctx = self.cublas_ctx(self.dsize)

    def _get_gmres(self):
        '''
        Get GMRES function pointer
        '''
        gmresfunc = kryit.gmres
        gmresfunc.argtypes = [c_uint]
        gmresfunc.restype  = c_void_p
        self.gmresfunc = gmresfunc
        self.gmres = self.gmresfunc(self.dsize)

    def _get_solver(self):
        '''
        Interface to solve the problem
        Feed the application context to the solver

        param 1: function pointer to mat dot vec approximation
        param 2: function pointer to preconditioner function
        param 3: function pointer to the gmres function instance
        param 4: pointer to solver app context
        param 5: pointer to preconditioner app context
        param 6: pointer to the gmres app context
        param 7: pointer to the cublas app context
        '''
        gmres_solver = kryit.gmres_solve
        gmres_solver.argtypes = [c_void_p, c_void_p, c_void_p,
                                 py_object, c_void_p, c_void_p, c_void_p]
        gmres_solver.restype = None
        self.solver = gmres_solver

    def solve(self):
        '''
        Interface to do the actual solving of the linear system
        '''
        self.solver(
            MFmatvecapp, PMGPrecond, self.gmres,  # three function pointers
            self.solctx,
            self.pctx, self.gctx, self.bctx  # four app context pointers
        )

# create the solver for testing
testsol = TestSolver(250)
# create the gmres for testing
testgmres = KryItOnGPU(testsol)
testgmres.solve()
kryit.print_data(testsol.x, c_uint(testsol.xdim), c_uint(testsol.dsize))
