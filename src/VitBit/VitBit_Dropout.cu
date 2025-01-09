#include <curand_kernel.h>

__global__ void VitBit_Dropout(const int8_t* TC_input, int8_t* TC_output, 
                                const int* CC_input_int, int* CC_output_int, 
                                const float* CC_input_fp, float* CC_output_fp, 
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
        int idx_input = row * input_cols + col;

        int core_idx = idx + 32 * (idx/64 - idx/32);
        int core_idx_input = idx_input + 32 * (idx_input/64 - idx_input/32);

        if((idx / warpSize) % 2 == 0){
            curandStatePhilox4_32_10_t state;
            curand_init(clock64(), core_idx_input, 0, &state);

            float rand_val = curand_uniform(&state);
            if (rand_val < dropout_rate) {
                CC_output_int[core_idx] = 0;
            } else {
                CC_output_int[core_idx] = CC_input_int[core_idx_input];
            }
        }
        else {
            curandStatePhilox4_32_10_t state;
            curand_init(clock64(), core_idx_input, 0, &state);

            float rand_val = curand_uniform(&state);
            if (rand_val < dropout_rate) {
                CC_output_fp[core_idx] = 0;
            } else {
                CC_output_fp[core_idx] = CC_input_fp[core_idx_input];
            }
        }
    }
}