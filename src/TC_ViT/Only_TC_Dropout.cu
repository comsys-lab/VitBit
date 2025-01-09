#include <curand_kernel.h>

__global__ void Only_TC_Dropout(const int8_t* input, int8_t* output, float dropout_rate, int input_rows, int input_cols, int output_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < input_rows && col < input_cols) {
        curandStatePhilox4_32_10_t state;
        curand_init(clock64(), row * input_cols + col, 0, &state);

        float rand_val = curand_uniform(&state);
        if (rand_val < dropout_rate) {
            output[row * output_cols + col] = 0;
        } else {
            output[row * output_cols + col] = input[row * input_cols + col];
        }
    }
}