__global__ void TC_INT_Add(const int8_t* TC_input, const int8_t* TC_weight, int8_t* TC_output,
                            const int *CC_input_int, const int *CC_weight_int, int *CC_output_int,
                            int TC_width, int CC_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int i = 0; i < TC_width; i++)
        TC_output[idx + i] = TC_input[idx + i] + TC_weight[idx];
    for(int i = 0; i < CC_width; i++)
        CC_output_int[idx + i] = CC_input_int[idx + i] + CC_weight_int[idx];
}
