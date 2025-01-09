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

__global__ void TC_INT_Gelu(int8_t* TC_input, int8_t* TC_output, 
                            int* CC_input, int* CC_output, 
                            int TC_width, int CC_width, int channels) {
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
    } else if (row < channels && col >= TC_width && col < (TC_width + CC_width)) {
        int CC_col = col - TC_width;
        int idx = row * CC_width + CC_col;

        // ShiftGELU function for CC_input
        int Ip = CC_input[idx] + (CC_input[idx] >> 1) + (CC_input[idx] >> 3) + (CC_input[idx] >> 4);

        int I_exp = shift_exp(Ip);
        int I_exp_prime = shift_exp(-Ip);

        int I_div = int_div(I_exp, I_exp + I_exp_prime);

        CC_output[idx] = CC_input[idx] * I_div;
    }
}