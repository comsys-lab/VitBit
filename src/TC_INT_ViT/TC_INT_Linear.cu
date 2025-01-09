#include "Gemm_kernels.cuh"

#define TILE_WIDTH1 32
#define TILE_WIDTH2 24
__global__ void TC_INT_Linear(const int8_t* TC_input, const int8_t* TC_weight, int8_t* TC_output, 
                            const int *CC_input_int, const int *CC_weight_int, int *CC_output_int,
                            int TC_M, int TC_N, int TC_K,
                            int CC_M, int CC_N, int CC_K){

    int bid = blockIdx.x * gridDim.x + blockIdx.y;

    if((threadIdx.y < 1)) {
        TC_gemm_kernel(TC_input, TC_weight, TC_output, TC_M, TC_N, TC_K);
    }              
    else if(bid < ((CC_N/TILE_WIDTH2)*(CC_M/TILE_WIDTH1))) {
        CC_INT_gemm_kernel(CC_input_int, CC_weight_int, CC_output_int,
                            CC_M, CC_N, CC_K);
    }         
}