__global__ void TC_INT_FP_Normalization(const int8_t* TC_input, int8_t* TC_output, const int8_t* TC_gamma, const int8_t* TC_beta, 
                                    const int* CC_input_int, int* CC_output_int, const int* CC_gamma_int, const int* CC_beta_int, 
                                    const float* CC_input_fp, float* CC_output_fp, const float* CC_gamma_fp, const float* CC_beta_fp, 
                                    int channels, int TC_width, int CC_width_half) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int TC_idx = row * TC_width + col;
    int CC_idx = row * CC_width_half + col;

    int threadIndex = row * (gridDim.x * blockDim.x) + col;
    int total_TC_elements = channels * TC_width;
    int total_CC_elements = channels * CC_width_half;

     if (threadIndex < total_TC_elements) {
        int channel_idx = TC_idx / TC_width;
        int channel_start = channel_idx * TC_width;

        // Calculate mean and variance for the current channel
        int8_t sum = 0;
        int8_t sum_squared = 0;
        for (int i = 0; i < TC_width; ++i) {
            int input_idx = channel_start + i;
            int8_t val = TC_input[input_idx];
            sum += val;
            sum_squared += val * val;
        }
        int8_t mean = sum / TC_width;
        int8_t variance = sum_squared / TC_width - mean * mean;

        // Normalize and scale the data
        int8_t normalized_val = ((TC_input[TC_idx] - mean) * 1000000) / (sqrt(variance + 1e-5f) * 1000) + 500000;
        int8_t scaled_val = (TC_gamma[channel_idx] * normalized_val + TC_beta[channel_idx]) / 1000000;

        // Convert back to integer format
        TC_output[TC_idx] = scaled_val;
    }
    
    else if (threadIndex < total_TC_elements + total_CC_elements) {
        int idx = threadIndex - total_TC_elements;
        int core_idx = idx + 32 * (idx/64 - idx/32);
        int channel_idx = core_idx / CC_width_half;
        int channel_start = channel_idx * CC_width_half;

        // Calculate mean and variance for the current channel
        int sum_int = 0;
        int sum_squared_int = 0;
        float sum_fp = 0;
        float sum_squared_fp = 0;

        if((idx / warpSize) % 2 == 0){
            for (int i = 0; i < CC_width_half; ++i) {
                int input_idx = channel_start + i;
                int val = CC_input_int[input_idx];
                sum_int += val;
                sum_squared_int += val * val;
            }
            int mean = sum_int / CC_width_half;
            int variance = sum_squared_int / CC_width_half - mean * mean;

            // Normalize and scale the data
            int normalized_val = ((CC_input_int[CC_idx] - mean) * 1000000) / (sqrt(variance + 1e-5f) * 1000) + 500000;
            int scaled_val = (CC_gamma_int[channel_idx] * normalized_val + CC_beta_int[channel_idx]) / 1000000;

            // Convert back to integer format
            CC_output_int[CC_idx] = scaled_val;
        }
        else {
            for (int i = 0; i < CC_width_half; ++i) {
                int input_idx = channel_start + i;
                float val = CC_input_fp[input_idx];
                sum_fp += val;
                sum_squared_fp += val * val;
            }
            float mean = sum_fp / CC_width_half;
            float variance = sum_squared_fp / CC_width_half - mean * mean;

            // Normalize and scale the data
            float normalized_val = ((CC_input_fp[CC_idx] - mean) * 1000000) / (sqrt(variance + 1e-5f) * 1000) + 500000;
            float scaled_val = (CC_gamma_fp[channel_idx] * normalized_val + CC_beta_fp[channel_idx]) / 1000000;

            // Convert back to integer format
            CC_output_fp[CC_idx] = scaled_val;
        }
    }
}
