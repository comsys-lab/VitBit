#include <curand_kernel.h>

__global__ void TC_FP_Dropout(const int8_t* TC_input, int8_t* TC_output, 
                                const float* CC_input, float* CC_output, 
                                float dropout_rate, int TC_input_rows, int CC_input_rows, int input_cols, int output_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < TC_input_rows && col < input_cols) {
        int idx = row * input_cols + col;
        int out_idx = row * output_cols + col;

        curandStatePhilox4_32_10_t state;
        curand_init(clock64(), idx, 0, &state);

        float rand_val = curand_uniform(&state);
        if (rand_val < dropout_rate) {
            TC_output[out_idx] = 0;
        } else {
            TC_output[out_idx] = TC_input[idx];
        }
    } else if (row < TC_input_rows + CC_input_rows && col < input_cols) {
        int CC_row = row - TC_input_rows;
        int idx = CC_row * input_cols + col;
        int out_idx = CC_row * output_cols + col;

        curandStatePhilox4_32_10_t state;
        curand_init(clock64(), idx, 0, &state);

        float rand_val = curand_uniform(&state);
        if (rand_val < dropout_rate) {
            CC_output[out_idx] = 0;
        } else {
            CC_output[out_idx] = CC_input[idx];
        }
    }
}