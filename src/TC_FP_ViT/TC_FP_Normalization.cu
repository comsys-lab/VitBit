__global__ void TC_FP_Normalization(const int8_t* TC_input, int8_t* TC_output, const int8_t* TC_gamma, const int8_t* TC_beta, 
                                    const float* CC_input, float* CC_output, const float* CC_gamma, const float* CC_beta, 
                                    int channels, int TC_width, int CC_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int TC_idx = row * TC_width + col;
    int CC_idx = row * CC_width + col;

    int threadIndex = row * (gridDim.x * blockDim.x) + col;
    int total_TC_elements = channels * TC_width;
    int total_CC_elements = channels * CC_width;

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
        int CC_threadIndex = threadIndex - total_TC_elements;
        int channel_idx = CC_threadIndex / CC_width;
        int channel_start = channel_idx * CC_width;

        // Calculate mean and variance for the current channel
        float sum = 0;
        float sum_squared = 0;
        for (int i = 0; i < CC_width; ++i) {
            int input_idx = channel_start + i;
            float val = CC_input[input_idx];
            sum += val;
            sum_squared += val * val;
        }
        float mean = sum / CC_width;
        float variance = sum_squared / CC_width - mean * mean;

        // Normalize and scale the data
        float normalized_val = ((CC_input[CC_idx] - mean) * 1000000) / (sqrt(variance + 1e-5f) * 1000) + 500000;
        float scaled_val = (CC_gamma[channel_idx] * normalized_val + CC_beta[channel_idx]) / 1000000;

        // Convert back to integer format
        CC_output[CC_idx] = scaled_val;
    }
}
