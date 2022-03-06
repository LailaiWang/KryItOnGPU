#ifndef __PADE_EXPM__
#define __PADE_EXPM__
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cmath>
#include <vector>
#include <type_traits>
#include <iostream>
#include "util.cuh"

template<typename T>
T factorial(T input) {
	if (input == (T)0) return 1;
	else {
		return input * factorial(input - 1);
	}
}

template<typename T>
void eval_pade_coeff(unsigned int p, T* mcoef) {
	// when m=n, we only need one copy of the coeffients,
	// which will be sent to the constant ram on GPU side
	// R(m/n) = p_m/q_n
	// p_m = sum_j=0^m (p+q-j)!p!/(p+q)!j!(p-j)!( A)^j
	// q_n = sum_j=0^m (p+q-j)!q!/(p+q)!j!(q-j)!(-A)^j
	long int a = factorial<long int>(1);
	for (unsigned int i = 0; i <= p; i++) {
		mcoef[i]  = (T) factorial<T>((T)(p + p - i));
		mcoef[i] *= (T) factorial<T>((T)p);
		mcoef[i] /= (T) factorial<T>((T)(p + p));
		mcoef[i] /= (T) factorial<T>((T)i);
		mcoef[i] /= (T) factorial<T>((T)(p - i));
	}
}


template<typename T>
__global__ 
void fill_testing_matrix(T* mat, unsigned int dim) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dim * dim) return;
	unsigned int i = idx % dim;  /*no. of row*/
	unsigned int j = idx / dim;  /*no. of col*/
	if (j >= i) mat[j * dim + i] = (T)1;
	else mat[j * dim + i] = (T)0;
}

template<typename T>
void triInverse_wrapper(T* mat, 
                        T* buffer,
                        unsigned int dim, 
                        cublasHandle_t handle,
                        T** &Abuf,
                        T** &Ainvbuf) {
	int* info = (int*)(buffer + dim * dim);
	int* pivot = info + dim * dim;
	cudaMemcpy(Abuf, &mat, sizeof(T*), cudaMemcpyHostToDevice);
	cudaMemcpy(Ainvbuf, &buffer, sizeof(T*), cudaMemcpyHostToDevice);
	if constexpr (std::is_same<T, double>::value) {
		cublasDgetrfBatched(handle, dim, Abuf, dim, pivot, info, 1);
		cublasDgetriBatched(handle, dim, Abuf, dim, pivot, Ainvbuf, dim, info, 1);
	} else 
	if constexpr (std::is_same<T, float>::value) {
		cublasSgetrfBatched(handle, dim, Abuf, dim, pivot, info, 1);
		cublasSgetriBatched(handle, dim, Abuf, dim, pivot, Ainvbuf, dim, info, 1);
	}
}

template<typename T>
__global__
void set_identity(T* mat, unsigned int dim) {
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx >= dim * dim) return;
	unsigned int n = idx % dim;
	unsigned int m = idx / dim;
	if (m == n) mat[m * dim + n] = (T)1.0f;
	if (m != n) mat[m * dim + n] = (T)0.0f;
}

/*
 * With Krylov approximation, these mats are small mats, we can indulge ourselves of 
 * allocating or reusing the space from somewhere else for this.
 */
template<typename T>
void eval_matrix_exponential(T* mat,    // the matrix as well as the output 
	                         T* mcoef,  // pade coefficient
	                         T* buffer, // buffer for calculation
							 T** &Abuf,
						     T** &Ainvbuf,		
	                         unsigned int dim,     // dimension of the problem
							 unsigned int p, // degree of pade approximant
						     T* alpha,   // constant alpha  1
							 T* beta,    // constant beta   0
						     T* gamma,   // constant gamma -1
							 cublasHandle_t* handle
							) {   
	if(p == 6) {
		// I A1 A2 A3 A4 A5 A6
		// comp A  A2
		// then A3 A4 // these can be done at the same time
		// then A5 A6 // these can be done at the same time
		T* A1 = buffer + dim * dim * 0;
		T* A2 = buffer + dim * dim * 1;
		T* A3 = buffer + dim * dim * 2;
		T* A4 = buffer + dim * dim * 3;
		T* A5 = buffer + dim * dim * 4;
		T* A6 = buffer + dim * dim * 5;
		cudaMemcpy(A1, mat, sizeof(T) * dim * dim, cudaMemcpyDeviceToDevice);

		unsigned int nblocks = std::ceil((T)(dim * dim) / 256);
		
		if constexpr (std::is_same<T, double>::value) {
			cudaStream_t stream0, stream1;
			cublasGetStream(handle[0], &stream0);
			cublasGetStream(handle[1], &stream1);

			/*first compute A2*/
			cublasDgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A1, dim, A1, dim, beta, A2, dim);
			// after moving the mat value to A1, set mat to identity matrix
			set_identity<T> <<<nblocks, 256, 0, stream1>>> (mat, dim);
			cudaDeviceSynchronize();

			/*use default stream to compute A3 and A5*/
			cublasDgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A1, dim, A2, dim, beta, A3, dim);
			cublasDgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A3, dim, beta, A5, dim);
			/*use recieved stream to compute A4 A6*/
			cublasDgemm(handle[1], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A2, dim, beta, A4, dim);
			cublasDgemm(handle[1], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A4, dim, beta, A6, dim);
			/*synchronize different streams*/
			cudaDeviceSynchronize();

			/*first stream*/
			cublasDaxpy(handle[0], dim * dim, &mcoef[2], A2, 1, mat, 1); //accumulate the positive terms to I + a2*A2
			cublasDaxpy(handle[0], dim * dim, &mcoef[4], A4, 1, mat, 1); // accumulate I + a2*A2+a4*A4
			cublasDaxpy(handle[0], dim * dim, &mcoef[6], A6, 1, mat, 1); // accumulate I + a2*A2+a4*A4 + a6*A6
			/*second stream*/
			cublasDscal(handle[1], dim*dim, &mcoef[1], A1, 1);         // scalce a1*A1
			cublasDaxpy(handle[1], dim * dim, &mcoef[3], A3, 1, A1,1); // accumulate a1*A1+a3*A3
			cublasDaxpy(handle[1], dim * dim, &mcoef[5], A5, 1, A1,1); // accumulate a1*A1+a3*A3+a5*A5
			cudaDeviceSynchronize();

			cudaMemcpyAsync(A2, mat, sizeof(T) * dim * dim, cudaMemcpyDeviceToDevice, stream0);
			cudaMemcpyAsync(A3, mat, sizeof(T) * dim * dim, cudaMemcpyDeviceToDevice, stream1);
			// the positive terms + positive/negtive is the numerator stored in A2
			cublasDaxpy(handle[0], dim * dim, alpha, A1, 1, A2, 1); //A2 + A1
			// the positive term - positive/negtive is the denominator stored in A1
			cublasDaxpy(handle[1], dim * dim, gamma, A1, 1, A3, 1); //A3 - A1
			cudaDeviceSynchronize();
			// calculate the inverse of A3
			triInverse_wrapper<T>(A3, A4, dim, handle[0], Abuf, Ainvbuf);
			// calculate the pade approximant
			cublasDgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A4, dim, beta, mat, dim);
			cudaStreamSynchronize(stream0);
		} else 
		if constexpr (std::is_same<T, float>::value) {
			cudaStream_t stream0, stream1;
			cublasGetStream(handle[0], &stream0);
			cublasGetStream(handle[1], &stream1);

			/*first compute A2*/
			cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A1, dim, A1, dim, beta, A2, dim);
			// after moving the mat value to A1, set mat to identity matrix
			set_identity<T> <<<nblocks, 256, 0, stream1>>> (mat, dim);
			cudaDeviceSynchronize();

			/*use default stream to compute A3 and A5*/
			cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A1, dim, A2, dim, beta, A3, dim);
			cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A3, dim, beta, A5, dim);
			/*use recieved stream to compute A4 A6*/
			cublasSgemm(handle[1], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A2, dim, beta, A4, dim);
			cublasSgemm(handle[1], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A4, dim, beta, A6, dim);
			/*synchronize different streams*/
			cudaDeviceSynchronize();

			/*first stream*/
			cublasSaxpy(handle[0], dim * dim, &mcoef[2], A2, 1, mat, 1); //accumulate the positive terms to I + a2*A2
			cublasSaxpy(handle[0], dim * dim, &mcoef[4], A4, 1, mat, 1); // accumulate I + a2*A2+a4*A4
			cublasSaxpy(handle[0], dim * dim, &mcoef[6], A6, 1, mat, 1); // accumulate I + a2*A2+a4*A4 + a6*A6
			/*second stream*/
			cublasSscal(handle[1], dim * dim, &mcoef[1], A1, 1);       // scalce a1*A1
			cublasSaxpy(handle[1], dim * dim, &mcoef[3], A3, 1, A1,1); // accumulate a1*A1+a3*A3
			cublasSaxpy(handle[1], dim * dim, &mcoef[5], A5, 1, A1,1); // accumulate a1*A1+a3*A3+a5*A5
			cudaDeviceSynchronize();

			cudaMemcpyAsync(A2, mat, sizeof(T) * dim * dim, cudaMemcpyDeviceToDevice, stream0);
			cudaMemcpyAsync(A3, mat, sizeof(T) * dim * dim, cudaMemcpyDeviceToDevice, stream1);
			// the positive terms + positive/negtive is the numerator stored in A2
			cublasSaxpy(handle[0], dim * dim, alpha, A1, 1, A2, 1); //A2 + A1
			// the positive term - positive/negtive is the denominator stored in A1
			cublasSaxpy(handle[1], dim * dim, gamma, A1, 1, A3, 1); //A3 - A1
			cudaDeviceSynchronize();
			// calculate the inverse of A3
			triInverse_wrapper<T>(A3, A4, dim, handle[0], Abuf, Ainvbuf);
			// calculate the pade approximant
			cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A4, dim, beta, mat, dim);
			cudaStreamSynchronize(stream0);
		}
	} else 
	if (p == 8) {
		// I A1 A2 A3 A4 A5 A6 A7 A8
		// comp A  A2
		// then A3 A4 // these can be done at the same time
		// then A5 A6 // these can be done at the same time
		T* A1 = buffer + dim * dim * 0;
		T* A2 = buffer + dim * dim * 1;
		T* A3 = buffer + dim * dim * 2;
		T* A4 = buffer + dim * dim * 3;
		T* A5 = buffer + dim * dim * 4;
		T* A6 = buffer + dim * dim * 5;
		T* A7 = buffer + dim * dim * 6;
		T* A8 = buffer + dim * dim * 8;
		cudaMemcpy(A1, mat, sizeof(T) * dim * dim, cudaMemcpyDeviceToDevice);

		unsigned int nblocks = std::ceil((T)(dim * dim) / 256);
		
		if constexpr (std::is_same<T, double>::value) {
			cudaStream_t stream0, stream1;
			cublasGetStream(handle[0], &stream0);
			cublasGetStream(handle[1], &stream1);

			/*first compute A2*/
			cublasDgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A1, dim, A1, dim, beta, A2, dim);
			// after moving the mat value to A1, set mat to identity matrix
			set_identity<T> <<<nblocks, 256, 0, stream1>>> (mat, dim);
			cudaDeviceSynchronize();

			/*use default stream to compute A3 and A5*/
			cublasDgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A1, dim, A2, dim, beta, A3, dim);
			cublasDgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A3, dim, beta, A5, dim);
			cublasDgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A5, dim, beta, A7, dim);
			/*use recieved stream to compute A4 A6*/
			cublasDgemm(handle[1], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A2, dim, beta, A4, dim);
			cublasDgemm(handle[1], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A4, dim, beta, A6, dim);
			cublasDgemm(handle[1], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A6, dim, beta, A8, dim);
			/*synchronize different streams*/
			cudaDeviceSynchronize();

			/*first stream*/
			cublasDaxpy(handle[0], dim * dim, &mcoef[2], A2, 1, mat, 1); //accumulate the positive terms to I + a2*A2
			cublasDaxpy(handle[0], dim * dim, &mcoef[4], A4, 1, mat, 1); // accumulate I + a2*A2+a4*A4
			cublasDaxpy(handle[0], dim * dim, &mcoef[6], A6, 1, mat, 1); // accumulate I + a2*A2+a4*A4 + a6*A6
			cublasDaxpy(handle[0], dim * dim, &mcoef[8], A8, 1, mat, 1); // accumulate I + a2*A2+a4*A4 + a6*A6 + a8*A8
			/*second stream*/
			cublasDscal(handle[1], dim * dim, &mcoef[1], A1, 1);         // scalce a1*A1
			cublasDaxpy(handle[1], dim * dim, &mcoef[3], A3, 1, A1,1); // accumulate a1*A1+a3*A3
			cublasDaxpy(handle[1], dim * dim, &mcoef[5], A5, 1, A1,1); // accumulate a1*A1+a3*A3+a5*A5
			cublasDaxpy(handle[1], dim * dim, &mcoef[7], A7, 1, A1,1); // accumulate a1*A1+a3*A3+a5*A5+a7*A7
			cudaDeviceSynchronize();

			cudaMemcpyAsync(A2, mat, sizeof(T) * dim * dim, cudaMemcpyDeviceToDevice, stream0);
			cudaMemcpyAsync(A3, mat, sizeof(T) * dim * dim, cudaMemcpyDeviceToDevice, stream1);
			// the positive terms + positive/negtive is the numerator stored in A2
			cublasDaxpy(handle[0], dim * dim, alpha, A1, 1, A2, 1); //A2 + A1
			// the positive term - positive/negtive is the denominator stored in A1
			cublasDaxpy(handle[1], dim * dim, gamma, A1, 1, A3, 1); //A3 - A1
			cudaDeviceSynchronize();
			// calculate the inverse of A3
			triInverse_wrapper<T>(A3, A4, dim, handle[0], Abuf, Ainvbuf);
			// calculate the pade approximant
			cublasDgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A4, dim, beta, mat, dim);
			cudaStreamSynchronize(stream0);
		} else 
		if constexpr (std::is_same<T, float>::value) {
			cudaStream_t stream0, stream1;
			cublasGetStream(handle[0], &stream0);
			cublasGetStream(handle[1], &stream1);

			/*first compute A2*/
			cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A1, dim, A1, dim, beta, A2, dim);
			// after moving the mat value to A1, set mat to identity matrix
			set_identity<T> <<<nblocks, 256, 0, stream1>>> (mat, dim);
			cudaDeviceSynchronize();

			/*use default stream to compute A3 and A5*/
			cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A1, dim, A2, dim, beta, A3, dim);
			cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A3, dim, beta, A5, dim);
			cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A5, dim, beta, A7, dim);
			/*use recieved stream to compute A4 A6*/
			cublasSgemm(handle[1], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A2, dim, beta, A4, dim);
			cublasSgemm(handle[1], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A4, dim, beta, A6, dim);
			cublasSgemm(handle[1], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A6, dim, beta, A8, dim);
			/*synchronize different streams*/
			cudaDeviceSynchronize();

			/*first stream*/
			cublasSaxpy(handle[0], dim * dim, &mcoef[2], A2, 1, mat, 1); //accumulate the positive terms to I + a2*A2
			cublasSaxpy(handle[0], dim * dim, &mcoef[4], A4, 1, mat, 1); // accumulate I + a2*A2+a4*A4
			cublasSaxpy(handle[0], dim * dim, &mcoef[6], A6, 1, mat, 1); // accumulate I + a2*A2+a4*A4 + a6*A6
			cublasSaxpy(handle[0], dim * dim, &mcoef[8], A8, 1, mat, 1); // accumulate I + a2*A2+a4*A4 + a6*A6
			/*second stream*/
			cublasSscal(handle[1], dim * dim, &mcoef[1], A1, 1);       // scalce a1*A1
			cublasSaxpy(handle[1], dim * dim, &mcoef[3], A3, 1, A1,1); // accumulate a1*A1+a3*A3
			cublasSaxpy(handle[1], dim * dim, &mcoef[5], A5, 1, A1,1); // accumulate a1*A1+a3*A3+a5*A5
			cublasSaxpy(handle[1], dim * dim, &mcoef[7], A7, 1, A1,1); // accumulate a1*A1+a3*A3+a5*A5
			cudaDeviceSynchronize();

			cudaMemcpyAsync(A2, mat, sizeof(T) * dim * dim, cudaMemcpyDeviceToDevice, stream0);
			cudaMemcpyAsync(A3, mat, sizeof(T) * dim * dim, cudaMemcpyDeviceToDevice, stream1);
			// the positive terms + positive/negtive is the numerator stored in A2
			cublasSaxpy(handle[0], dim * dim, alpha, A1, 1, A2, 1); //A2 + A1
			// the positive term - positive/negtive is the denominator stored in A1
			cublasSaxpy(handle[1], dim * dim, gamma, A1, 1, A3, 1); //A3 - A1
			cudaDeviceSynchronize();
			// calculate the inverse of A3
			triInverse_wrapper<T>(A3, A4, dim, handle[0], Abuf, Ainvbuf);
			// calculate the pade approximant
			cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, A2, dim, A4, dim, beta, mat, dim);
			cudaStreamSynchronize(stream0);
		}

	}
}

template<typename T>
void alloc_expm_ram(unsigned int p, unsigned int dim, T* &buffer, T* &padecoef_d, char* &aby) {
	/*try to align our data*/
	unsigned int soasz = 32;
	unsigned int nbufs = 0;

    dim = dim*10;

	nbufs = soasz* std::ceil((T)((p + 1) * (dim * dim)) / soasz);
	cudaMalloc((void**)&buffer, sizeof(T) * nbufs);
	nbufs = soasz*std::ceil((T)((p+1))/soasz);
	cudaMalloc((void**)&padecoef_d, sizeof(T) * nbufs);
	/*8 double precision floats*/
	cudaMalloc((void**)&aby, sizeof(char) * 64);
}

template<typename T>
void clean_expm_ram(T* buffer, T* padecoef_d, char* aby) {
	cudaFree(buffer);
	cudaFree(padecoef_d);
	cudaFree(aby);
}

/*define the ram needed separately for reuse purpose*/
template<typename T>
struct expm_ctx_ram {
	// host 
	std::vector<T> padecoef;
	unsigned int p = 0;
	unsigned int dim = 0;
	//device
	T* padecoef_d;
	T* buffer;
	char* aby;

	void (*_eval_pade_coeff) (unsigned int, T*);
	void (*_alloc_expm_ram) (unsigned int, unsigned int, T* &, T* &, char* &);
	void (*_clean_expm_ram) (T*, T*, char*);

	std::vector<T> (*_view_content) (T* input, unsigned int dims);

	expm_ctx_ram(unsigned int pin, unsigned int din) {
		p   = pin;
		dim = din;
		padecoef.resize(p+1);

		_eval_pade_coeff = &eval_pade_coeff<T>;
		_alloc_expm_ram  = &alloc_expm_ram<T>;
		_clean_expm_ram  = &clean_expm_ram<T>;
		_view_content = &view_content<T>;
		
		/*start working on some stuff*/
		_eval_pade_coeff(p, padecoef.data());
		_alloc_expm_ram(p, dim, buffer, padecoef_d, aby);
		
		/*copy the coefficients to device*/
		cudaMemcpy(padecoef_d, padecoef.data(), sizeof(T) * (p + 1), cudaMemcpyHostToDevice);
		
		/*we have space for 64 bytes, 8 double floats*/
		T alps[] = { (T)1,(T)0,(T)-1 };
		cudaMemcpy((T*)aby, alps, sizeof(T) * 3, cudaMemcpyHostToDevice);
	}
	
	~expm_ctx_ram() {
		_clean_expm_ram(buffer, padecoef_d, aby);
		_eval_pade_coeff = nullptr;
		_alloc_expm_ram  = nullptr;
		_clean_expm_ram  = nullptr;
	}

};

template<typename T>
struct expm_ctx {
	unsigned int dim; // dimension of the problem
	unsigned int p;   // polynomial degree 
	cublasHandle_t handle[2]; // two handles
	
	// define some function pointers here
	void (*_set_identity) (T*, unsigned int);
	void (*_eval_matrix_exponential) (T*, T*, T*,
								      T** &, T** &,
									  unsigned int, unsigned int,
		                              T*, T*, T*, cublasHandle_t*);

	struct expm_ctx_ram<T> *ram_ctx;
	
	T** A;
	T** Ainv;

	/*write a small constructor here*/
	expm_ctx(unsigned int pin, unsigned int din, void* ram_ctx_in, cudaStream_t streamin) {
		p   = pin;
		dim = din;
		ram_ctx = (struct expm_ctx_ram<T>*) ram_ctx_in;
		/*instantiate functions and function ptrs*/
		_set_identity = &set_identity<T>;
		_eval_matrix_exponential = &eval_matrix_exponential<T>;
		/*create two handles*/
		cublasCreate(&handle[0]);
		cublasCreate(&handle[1]);

		/*set the pointer mode for cublas, use device mode for sure*/
		cublasSetPointerMode(handle[0], CUBLAS_POINTER_MODE_DEVICE);
		cublasSetPointerMode(handle[1], CUBLAS_POINTER_MODE_DEVICE);

		/*first one use the assigned stream, second one use the null stream*/
		cublasSetStream(handle[0], streamin);
		cublasSetStream(handle[1], 0);

		cudaStream_t s1; cublasGetStream(handle[0], &s1);
		cudaStream_t s2; cublasGetStream(handle[1], &s2);

		/*Allocate A and Ainv on Device*/
		cudaMalloc((void**)&A, sizeof(T*));
		cudaMalloc((void**)&Ainv, sizeof(T*));
	}

	/*a small destructor here*/
	~expm_ctx() {
		cublasDestroy(handle[0]);
		cublasDestroy(handle[1]);

		_set_identity = nullptr;
		_eval_matrix_exponential = nullptr;
		ram_ctx = nullptr;
		cudaFree(A);
		cudaFree(Ainv);
	}

};
#endif
