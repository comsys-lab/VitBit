#include "Gemm_kernels.cuh"

#define TILE_WIDTH1 32
#define TILE_WIDTH2 24
__global__ void TC_INT_FP_Linear(const int8_t* TC_input, const int8_t* TC_weight, int8_t* TC_output, 
                            const int *CC_input_int, const int *CC_weight_int, int *CC_output_int,
                            const float *CC_input_fp, const float *CC_weight_fp, float *CC_output_fp,
                            int TC_M, int TC_N, int TC_K,
                            int CC_M, int CC_N_half, int CC_K){

    int bid = blockIdx.x * gridDim.x + blockIdx.y;

    if((threadIdx.y < 1)) {
        TC_gemm_kernel(TC_input, TC_weight, TC_output, TC_M, TC_N, TC_K);
    }              
    else if(bid < ((CC_N_half/TILE_WIDTH2)*(CC_M/TILE_WIDTH1))) {
        CC_INT_FP_gemm_kernel(CC_input_int, CC_weight_int, CC_output_int,
                            CC_input_fp, CC_weight_fp, CC_output_fp,
                            CC_M, CC_N_half, CC_K);
    }         
}