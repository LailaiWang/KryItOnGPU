#ifndef __CUDA__CONSTANTS_CUH__
#define __CUDA__CONSTANTS_CUH__

__constant__ float  P_ONE_F = 1.0f;
__constant__ float  N_ONE_F =-1.0f;
__constant__ double P_ONE_D = 1.0f; 
__constant__ double N_ONE_D =-1.0f; 

/*these can be accessed by device code*/
const float  P_1F = 1.0f;
const float  N_1F =-1.0f;
const double P_1D = 1.0f; 
const double N_1D =-1.0f; 
#endif
