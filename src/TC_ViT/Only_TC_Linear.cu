#include "Gemm_kernels.cuh"

__global__ void Only_TC_Linear(const int8_t* TC_input, const int8_t* TC_weight, int8_t* TC_output, 
                                int TC_M, int TC_N, int TC_K){
    if((threadIdx.y < 1)) {
        TC_gemm_kernel(TC_input, TC_weight, TC_output, TC_M, TC_N, TC_K);
    } 
}
