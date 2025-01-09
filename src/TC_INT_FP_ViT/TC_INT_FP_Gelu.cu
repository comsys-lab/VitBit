__device__ int shift_exp(int input) {
    int exponent = 0;
    while ((input >> exponent) > 1) {
        exponent++;
    }
    return input - (1 << exponent); 
}

__device__ int int_div(int numerator, int denominator) {
    return (numerator + (1 << 31)) / denominator;
}

__global__ void TC_INT_FP_Gelu(int8_t* TC_input, int8_t* TC_output, 
                            int* CC_input_int, int* CC_output_int, 
                            float* CC_input_fp, float* CC_output_fp, 
                            int TC_width, int CC_width_half, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < channels && col < TC_width) {
        int idx = row * TC_width + col;
        
        // ShiftGELU function for TC_input
        int8_t Ip = TC_input[idx] + (TC_input[idx] >> 1) + (TC_input[idx] >> 3) + (TC_input[idx] >> 4);

        int I_exp = shift_exp(Ip);
        int I_exp_prime = shift_exp(-Ip);

        int I_div = int_div(I_exp, I_exp + I_exp_prime);

        TC_output[idx] = TC_input[idx] * I_div;
    } else if (row < channels && col >= TC_width && col < (TC_width + CC_width_half)) {
        int CC_col = col - TC_width;
        int idx = row * CC_width_half + CC_col;
        int core_idx = idx + 32 * (idx/64 - idx/32);

        if((idx / warpSize) % 2 == 0){
        int Ip = CC_input_int[idx] + (CC_input_int[idx] >> 1) + (CC_input_int[idx] >> 3) + (CC_input_int[idx] >> 4);

        int I_exp;
        I_exp = shift_exp(0);
        int I_exp_prime;
        I_exp_prime = shift_exp(-Ip);

        int I_div = int_div(I_exp, I_exp + I_exp_prime);

        CC_output_int[idx] = CC_input_int[idx] * I_div;
        }
        else{
            CC_output_fp[core_idx] =  0.5f * CC_input_fp[core_idx] * (1.0f + tanh(sqrtf(2.0f / M_PI)
                                * (CC_input_fp[core_idx] + 0.044715f * CC_input_fp[core_idx] * CC_input_fp[core_idx] * CC_input_fp[core_idx])));
        }
    }
}