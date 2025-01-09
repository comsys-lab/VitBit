// __global__ void Gelu_INT(const int* input, int* output, int width) {
    
//     int row = blockIdx.y * blockDim.y + threadIdx.y; 
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int idx = row * width + col;
        
//     output[idx] = max(0, input[idx]);
// }


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

__global__ void Only_TC_Gelu(int8_t* input, int8_t* output, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = row * width + col;

    // ShiftGELU function
    int8_t Ip = input[idx] + (input[idx] >> 1) + (input[idx] >> 3) + (input[idx] >> 4);

    int I_exp;
    I_exp = shift_exp(0);
    int I_exp_prime;
    I_exp_prime = shift_exp(-Ip);

    int I_div = int_div(I_exp, I_exp + I_exp_prime);

    output[idx] = input[idx] * I_div;
}