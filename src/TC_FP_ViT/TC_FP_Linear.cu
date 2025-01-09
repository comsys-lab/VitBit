#include "Gemm_kernels.cuh"

#define TILE_WIDTH1 32
#define TILE_WIDTH2 24
__global__ void TC_FP_Linear(const int8_t* TC_input, const int8_t* TC_weight, int8_t* TC_output, 
                            const float *CC_input_float, const float *CC_weight_float, float *CC_output_float,
                            int TC_M, int TC_N, int TC_K,
                            int CC_M, int CC_N, int CC_K){

    int bid = blockIdx.x * gridDim.x + blockIdx.y;

    if((threadIdx.y < 1)) {
        TC_gemm_kernel(TC_input, TC_weight, TC_output, TC_M, TC_N, TC_K);
    }              
    else if(bid < ((CC_N/TILE_WIDTH2)*(CC_M/TILE_WIDTH1))) {
        CC_FP_gemm_kernel(CC_input_float, CC_weight_float, CC_output_float,
                            CC_M, CC_N, CC_K);
    }         
}